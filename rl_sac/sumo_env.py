from __future__ import annotations

import os
import random
import subprocess
import sys
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

os.environ.setdefault("LIBSUMO", "1")

try:
    import traci  # type: ignore
    from sumolib import checkBinary  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "SUMO's Python tools (traci, sumolib) must be on PYTHONPATH. Set SUMO_HOME before running."
    ) from exc

from .config import EnvConfig


def _set_sumo_home_from_env() -> None:
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home and sumo_home not in sys.path:
        sys.path.append(os.path.join(sumo_home, "tools"))


class CavSpeedEnv:
    """TraCI-based environment for SAC speed control on the east-west mainline CAVs."""

    def __init__(self, config: EnvConfig | None = None, use_gui: bool = False) -> None:
        _set_sumo_home_from_env()
        self.config = config or EnvConfig()
        self.use_gui = use_gui
        self._conn: traci.connection.Connection | None = None
        self._conn_label: str | None = None
        self._proc: subprocess.Popen | None = None
        self._using_libsumo = os.environ.get("LIBSUMO") == "1"
        self._lane_lengths: Dict[str, float] = {}
        self._lane_capacities: Dict[str, float] = {}
        self._max_queue_capacity: float = 1.0
        self._history: deque[np.ndarray] = deque(maxlen=self.config.observation_history)
        self._last_actions = np.zeros(len(self.config.controlled_lanes), dtype=np.float32)
        self._last_energy = 0.0
        self._step = 0
        self._rng = random.Random(self.config.seed)
        self._obs_template = np.zeros(self._frame_dim, dtype=np.float32)
        self._last_arrived = 0
        self._lane_target_speeds = np.full(len(self.config.controlled_lanes), self.config.action_high, dtype=np.float32)
        self._prev_speed_targets = self._lane_target_speeds.copy()
        self._last_phase_index = 0
        self._episode_idx = 0
        self._cumulative_energy_kwh = 0.0
        self._cumulative_wait_all = 0.0
        self._cumulative_queue_all = 0.0
        self._phase_color_ids = {"green": 0, "yellow": 1, "red": 2}
        self._last_phase_color = "green"
        self._last_phase_remaining = 0.0
        self._episode_tag = "ep00000"

    # ------------------------------ properties -----------------------------

    @property
    def action_dim(self) -> int:
        return len(self.config.controlled_lanes)

    @property
    def frame_features_per_lane(self) -> int:
        return 6  # density, cav_share, speed, halting_ratio, wait, last_action

    @property
    def _frame_dim(self) -> int:
        per_lane = len(self.config.controlled_lanes) * self.frame_features_per_lane
        # plus corridor-level stats (3), mean corridor speed (1), progress (1),
        # phase one-hot (phase_bins) and phase progress + remaining (2)
        return per_lane + 5 + self.config.phase_bins + 2

    @property
    def observation_dim(self) -> int:
        return self._frame_dim * self.config.observation_history

    # ------------------------------ public api -----------------------------

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.close()
        if seed is not None:
            self._rng.seed(seed)
        self._episode_idx += 1
        self._episode_tag = f"ep{self._episode_idx:05d}"
        self._cumulative_energy_kwh = 0.0
        self._cumulative_wait_all = 0.0
        self._cumulative_queue_all = 0.0
        self._last_phase_color = "green"
        self._last_phase_remaining = 0.0
        self._start_traci()
        self._step = 0
        self._last_actions = np.zeros(self.action_dim, dtype=np.float32)
        self._last_arrived = 0
        self._history.clear()
        self._prime_history()
        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        if self._conn is None:
            raise RuntimeError("Environment must be reset before stepping.")
        clipped = np.clip(action, -1.0, 1.0).astype(np.float32)
        target_speeds = self._scale_actions_to_speed(clipped)
        applied_speeds = self._apply_speed_commands(target_speeds)
        self._conn.simulationStep()
        self._step += 1
        frame = self._collect_frame(clipped)
        self._history.append(frame)
        reward, reward_terms = self._compute_reward(frame)
        done = self._step >= self.config.max_steps or self._conn.simulation.getMinExpectedNumber() <= 0
        obs = self._get_observation()
        info = {
            **reward_terms,
            "step": float(self._step),
            "mean_action_speed": float(applied_speeds.mean()) if len(applied_speeds) else 0.0,
            "phase_bin": float(self._phase_bin()),
            "phase_color_id": float(self._phase_color_ids.get(self._last_phase_color, -1)),
            "phase_remaining": self._last_phase_remaining,
        }
        self._last_actions = clipped
        return obs, reward, done, info

    def close(self) -> None:
        if self._conn is not None and not self._using_libsumo:
            try:
                self._conn.close()
            except getattr(traci, "TraCIException", Exception):
                pass
        elif self._conn_label is not None and not self._using_libsumo:
            try:
                traci.getConnection(self._conn_label).close()
            except Exception:
                pass
        elif self._using_libsumo:
            try:
                traci.close()
            except Exception:
                pass
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None
        self._conn = None
        self._conn_label = None

    # --------------------------- setup & teardown --------------------------

    def _start_traci(self) -> None:
        cfg = self.config
        cmd = [checkBinary(cfg.sumo_gui_binary if self.use_gui else cfg.sumo_binary)]
        cmd += ["-c", str(cfg.sumo_config.resolve())]
        cmd += ["--step-length", str(cfg.step_length)]
        cmd += ["--quit-on-end"]
        cmd += ["--time-to-teleport", "-1"]
        cmd += ["--seed", str(self._rng.randint(0, 1_000_000))]
        cmd += ["--no-warnings"]
        cmd += ["--xml-validation", "never"]
        cmd += ["--duration-log.disable", "true"]
        cmd += ["--log", "NUL"] if os.name == "nt" else ["--log", "/dev/null"]
        cmd += ["--error-log", "NUL"] if os.name == "nt" else ["--error-log", "/dev/null"]
        self._append_output_args(cmd)
        label = f"cav-sac-{cfg.port}"
        self._conn_label = label
        if self._using_libsumo:
            traci.start(cmd)
            self._conn = traci
            self._proc = None
        else:
            result = traci.start(cmd, port=cfg.port, label=label)
            proc: subprocess.Popen | None = None
            if isinstance(result, tuple):
                proc = result[0]
            self._proc = proc
            self._conn = traci.getConnection(label)
        self._lane_lengths = {
            lane_id: self._conn.lane.getLength(lane_id)
            for lane_id in cfg.controlled_lanes
        }
        self._lane_capacities = {
            lane_id: max(length / max(self.config.density_headway, 1.0), 1.0)
            for lane_id, length in self._lane_lengths.items()
        }
        self._max_queue_capacity = sum(self._lane_capacities.values())
        self._lane_target_speeds = np.full(len(self.config.controlled_lanes), self.config.action_high, dtype=np.float32)
        self._prev_speed_targets = self._lane_target_speeds.copy()
        self._warmup()

    def _warmup(self) -> None:
        if self._conn is None:
            return
        for _ in range(self.config.warmup_steps):
            self._conn.simulationStep()

    def _append_output_args(self, cmd: List[str]) -> None:
        queue_dir = self._maybe_output_dir(self.config.queue_output_dir, "queue")
        tripinfo_dir = self._maybe_output_dir(self.config.tripinfo_output_dir, "tripinfo")
        fcd_dir = self._maybe_output_dir(self.config.fcd_output_dir, "fcd")
        summary_dir = self._maybe_output_dir(self.config.summary_output_dir, "summary")
        queue_path = self._resolve_output_path(queue_dir, "queue")
        tripinfo_path = self._resolve_output_path(tripinfo_dir, "tripinfo")
        fcd_path = self._resolve_output_path(fcd_dir, "fcd")
        summary_path = self._resolve_output_path(summary_dir, "summary")
        if queue_path:
            cmd += ["--queue-output", queue_path]
        if tripinfo_path:
            cmd += ["--tripinfo-output", tripinfo_path]
        if fcd_path:
            cmd += ["--fcd-output", fcd_path]
        if summary_path:
            cmd += ["--summary-output", summary_path]

    def _maybe_output_dir(self, configured: Path | None, default_name: str) -> Path | None:
        if configured is not None:
            return Path(configured)
        if self.config.sumo_output_root is None:
            return None
        return Path(self.config.sumo_output_root) / default_name

    def _resolve_output_path(self, base: Path | None, prefix: str, extension: str = ".xml") -> str | None:
        if base is None:
            return None
        base_path = Path(base)
        if base_path.suffix:
            target = base_path
        else:
            target = base_path / f"{prefix}_{self._episode_tag}{extension}"
        target.parent.mkdir(parents=True, exist_ok=True)
        return str(target)

    # -------------------------- observation helpers ------------------------

    def _prime_history(self) -> None:
        frame = self._collect_frame(self._last_actions)
        for _ in range(self.config.observation_history):
            self._history.append(frame.copy())

    def _get_observation(self) -> np.ndarray:
        if not self._history:
            self._prime_history()
        return np.concatenate(list(self._history)).astype(np.float32)

    def _collect_frame(self, actions: np.ndarray) -> np.ndarray:
        lane_metrics: List[float] = []
        total_cav = 0
        total_flow = 0
        total_wait = 0.0
        queue_veh = 0
        cfg = self.config
        phase_one_hot, phase_progress, phase_idx, phase_remaining = self._phase_features()
        for idx, lane_id in enumerate(cfg.controlled_lanes):
            vehicles = self._conn.lane.getLastStepVehicleIDs(lane_id)
            len_lane = self._lane_lengths.get(lane_id, 1.0)
            speeds = []
            waits = []
            cav_count = 0
            for vid in vehicles:
                if self._is_cav(vid):
                    cav_count += 1
                    speeds.append(self._conn.vehicle.getSpeed(vid))
                    waits.append(self._conn.vehicle.getWaitingTime(vid))
            total_cav += cav_count
            total_flow += len(vehicles)
            total_wait += sum(waits)
            _, total_halt = self._count_halting(vehicles)
            queue_veh += total_halt
            capacity = self._lane_capacities.get(lane_id, max(len_lane / max(cfg.density_headway, 1.0), 1.0))
            density = min(len(vehicles) / max(capacity, 1.0), 1.0)
            cav_share = cav_count / max(len(vehicles), 1)
            mean_speed = (sum(speeds) / len(speeds)) if speeds else 0.0
            wait_avg = (sum(waits) / len(waits)) if waits else 0.0
            halting_ratio = total_halt / max(len(vehicles), 1)
            lane_metrics.extend([
                density,
                cav_share,
                min(mean_speed / max(cfg.action_high, 1e-3), 1.0),
                halting_ratio,
                min(wait_avg / max(cfg.wait_normalizer, 1e-3), 1.0),
                float(actions[idx]) if idx < len(actions) else 0.0,
            ])
        throughput = float(self._conn.simulation.getArrivedNumber())
        progress = self._step / max(self.config.max_steps, 1)
        corridor_speed = self._mean_corridor_speed()
        remaining_norm = min(phase_remaining / max(cfg.phase_cycle, cfg.step_length), 1.0)
        features = (
            lane_metrics
            + [
                min(total_cav / max(cfg.corridor_cav_normalizer, 1.0), 1.0),
                min(total_flow / max(cfg.corridor_flow_normalizer, 1.0), 1.0),
                min(total_wait / max(cfg.corridor_wait_normalizer, 1.0), 1.0),
                min(corridor_speed / max(cfg.corridor_speed_normalizer, 1e-3), 1.0),
            ]
            + [progress]
            + phase_one_hot.tolist()
            + [phase_progress, remaining_norm]
        )
        return np.array(features, dtype=np.float32)

    def _current_phase_index(self) -> int:
        if self._conn is None:
            return 0
        try:
            return int(self._conn.trafficlight.getPhase(self.config.phase_signal_id))
        except traci.TraCIException:
            return 0

    def _phase_features(self) -> Tuple[np.ndarray, float, int, float]:
        cfg = self.config
        phase_idx = self._current_phase_index()
        one_hot = np.zeros(cfg.phase_bins, dtype=np.float32)
        one_hot[min(phase_idx, cfg.phase_bins - 1)] = 1.0
        progress = 0.0
        remaining = 0.0
        if self._conn is not None:
            try:
                duration = max(self._conn.trafficlight.getPhaseDuration(cfg.phase_signal_id), cfg.step_length)
                next_switch = self._conn.trafficlight.getNextSwitch(cfg.phase_signal_id)
                sim_time = self._conn.simulation.getTime()
                remaining = max(next_switch - sim_time, 0.0)
                progress = 1.0 - min(remaining / duration, 1.0)
            except traci.TraCIException:
                progress = 0.0
        self._last_phase_remaining = remaining
        return one_hot, progress, phase_idx, remaining

    def _phase_category(self, phase_idx: int) -> str:
        cfg = self.config
        if phase_idx in cfg.mainline_green_phases:
            return "green"
        if phase_idx in cfg.mainline_yellow_phases:
            return "yellow"
        return "red"

    def _mean_corridor_speed(self) -> float:
        speed_sum = 0.0
        count = 0
        for lane_id in self.config.controlled_lanes:
            try:
                speed = self._conn.lane.getLastStepMeanSpeed(lane_id)
            except traci.TraCIException:
                speed = 0.0
            speed_sum += speed
            count += 1
        return speed_sum / max(count, 1)

    def _phase_bin(self) -> int:
        cycle = max(self.config.phase_cycle, self.config.step_length)
        bins = max(self.config.phase_bins, 1)
        sim_time = self._conn.simulation.getTime() if self._conn else 0.0
        bin_width = cycle / bins
        return int((sim_time % cycle) // bin_width) % bins

    # -------------------------- action & control ---------------------------

    def _scale_actions_to_speed(self, actions: np.ndarray) -> np.ndarray:
        cfg = self.config
        half_range = (cfg.action_high - cfg.action_low) / 2.0
        return cfg.action_low + (actions + 1.0) * half_range

    def _lane_speed_limit(self, lane_id: str) -> float:
        limit = self.config.action_high
        if self._conn is not None:
            try:
                limit = self._conn.lane.getMaxSpeed(lane_id)
            except traci.TraCIException:
                limit = self.config.action_high
        return limit * self.config.speed_limit_scale

    def _stopline_speed_profile(self, base_target: float, distance_to_stop: float, phase_color: str) -> float:
        cfg = self.config
        if phase_color == "green":
            return max(0.0, min(base_target, cfg.action_high))
        if phase_color == "yellow":
            buffer = max(cfg.stopline_buffer_yellow, 1.0)
            cap = cfg.yellow_phase_speed_cap
        else:
            buffer = max(cfg.stopline_buffer, 1.0)
            cap = cfg.red_phase_speed_cap
        scaled_cap = cap
        if distance_to_stop <= buffer:
            scaled_cap = cap * max(distance_to_stop / buffer, 0.0)
        return max(0.0, min(base_target, scaled_cap))

    def _apply_speed_commands(self, lane_targets: np.ndarray) -> np.ndarray:
        cfg = self.config
        applied = np.zeros_like(lane_targets)
        self._prev_speed_targets = self._lane_target_speeds.copy()
        phase_idx = self._current_phase_index()
        phase_color = self._phase_category(phase_idx)
        if phase_color == "green":
            phase_scale = cfg.phase_speed_scale_green
        elif phase_color == "yellow":
            phase_scale = cfg.phase_speed_scale_yellow
        else:
            phase_scale = cfg.phase_speed_scale_red
        phase_scale = float(np.clip(phase_scale, 0.0, 1.0))
        phase_high = cfg.action_low + (cfg.action_high - cfg.action_low) * phase_scale
        phase_high = max(cfg.action_low, phase_high)
        self._last_phase_color = phase_color
        for lane_idx, lane_id in enumerate(cfg.controlled_lanes):
            lane_cap = min(phase_high, self._lane_speed_limit(lane_id))
            desired = float(np.clip(lane_targets[lane_idx], cfg.action_low, lane_cap))
            last = self._lane_target_speeds[lane_idx]
            max_delta = max(cfg.max_speed_delta, 1e-3)
            limited = float(np.clip(desired, last - max_delta, last + max_delta))
            limited = min(limited, lane_cap)
            self._lane_target_speeds[lane_idx] = limited
            applied[lane_idx] = limited
            vehicles = self._conn.lane.getLastStepVehicleIDs(lane_id)
            for vid in vehicles:
                if not self._is_cav(vid):
                    continue
                try:
                    lane_index = self._conn.vehicle.getLaneIndex(vid)
                except traci.TraCIException:
                    continue
                if lane_index != 1:
                    continue
                try:
                    lane_pos = self._conn.vehicle.getLanePosition(vid)
                except traci.TraCIException:
                    lane_pos = 0.0
                lane_length = self._lane_lengths.get(lane_id, 0.0)
                distance_to_stop = max(lane_length - lane_pos, 0.0)
                vehicle_target = self._stopline_speed_profile(limited, distance_to_stop, phase_color)
                vehicle_target = min(vehicle_target, lane_cap)
                try:
                    self._conn.vehicle.setSpeedMode(vid, cfg.cav_speed_mode)
                    self._conn.vehicle.setSpeed(vid, vehicle_target)
                except traci.TraCIException:
                    continue
        return applied

    # ---------------------------- reward logic -----------------------------

    def _compute_reward(self, frame: np.ndarray) -> Tuple[float, Dict[str, float]]:
        cfg = self.config
        weights = cfg.reward_weights
        (
            avg_power_kw,
            avg_energy_kwh,
            total_energy_kwh,
            cav_count,
        ) = self._estimate_energy_usage()
        self._cumulative_energy_kwh += total_energy_kwh
        cav_queue, total_queue = self._total_halting()
        cav_wait, total_wait = self._total_waiting_time()
        hdv_wait = max(total_wait - cav_wait, 0.0)
        self._cumulative_queue_all += total_queue
        self._cumulative_wait_all += total_wait
        arrived_total = self._conn.simulation.getArrivedNumber()
        arrived_delta = max(arrived_total - self._last_arrived, 0)
        self._last_arrived = arrived_total
        capacity = max(self._max_queue_capacity, 1.0)
        queue_ratio = min(total_queue / capacity, 1.0)
        queue_ratio_cav = min(cav_queue / capacity, 1.0)
        energy_norm = min(avg_power_kw / max(cfg.energy_normalizer, 1e-3), 1.0)
        cumulative_energy_norm = min(
            self._cumulative_energy_kwh
            / max(cfg.energy_normalizer * max(self._step + 1, 1) * cfg.step_length / 3600.0, 1e-3),
            1.0,
        )
        wait_norm = min(cav_wait / max(cfg.corridor_wait_normalizer, 1.0), 1.0)
        hdv_wait_norm = min(hdv_wait / max(cfg.corridor_wait_normalizer, 1.0), 1.0)
        throughput_norm = min(arrived_delta / max(cfg.throughput_normalizer, 1.0), 1.0)
        speed_delta = (self._lane_target_speeds - self._prev_speed_targets) / max(cfg.max_speed_delta, 1e-3)
        smooth = float(np.mean(np.square(speed_delta)))
        _, phase_progress, phase_idx, _ = self._phase_features()
        phase_color = self._phase_category(phase_idx)
        is_green = phase_color == "green"
        is_yellow = phase_color == "yellow"
        throughput_green = throughput_norm if is_green else 0.0
        green_window = 0.0
        if is_green and phase_progress <= cfg.green_wave_window:
            green_window = (1.0 - queue_ratio) * throughput_norm
        phase_penalty_term = queue_ratio
        if is_green:
            phase_penalty_term *= (1.0 - phase_progress)
        elif is_yellow:
            phase_penalty_term *= 0.8
        else:
            phase_penalty_term *= 1.2
        red_queue_ratio = queue_ratio if phase_color == "red" else 0.0

        reward = (
            -weights.energy * energy_norm
            -0.5 * weights.energy * cumulative_energy_norm
            -weights.queue * queue_ratio
            -weights.wait * wait_norm
            -weights.hdv_wait * hdv_wait_norm
            -weights.smooth * smooth
            +weights.throughput * throughput_green
            +weights.green_wave * green_window
            -weights.phase_penalty * phase_penalty_term
        )
        terms = {
            "energy": avg_power_kw,
            "energy_norm": energy_norm,
            "energy_kwh_step": avg_energy_kwh,
            "energy_kwh_total": total_energy_kwh,
            "energy_cum_norm": cumulative_energy_norm,
            "queue": float(total_queue),
            "queue_ratio": float(queue_ratio),
            "queue_cav": float(cav_queue),
            "queue_ratio_cav": float(queue_ratio_cav),
            "wait_time": float(cav_wait),
            "wait_norm": wait_norm,
            "wait_all": float(total_wait),
            "wait_hdv": float(hdv_wait),
            "wait_hdv_norm": hdv_wait_norm,
            "smooth": smooth,
            "throughput": float(arrived_delta),
            "throughput_norm": throughput_norm,
            "throughput_green": throughput_green,
            "green_window_reward": green_window,
            "phase_penalty_term": phase_penalty_term,
            "red_queue_ratio": float(red_queue_ratio),
            "phase_color": phase_color,
            "cumulative_energy_kwh": self._cumulative_energy_kwh,
            "cumulative_queue": self._cumulative_queue_all,
            "cumulative_wait": self._cumulative_wait_all,
            "reward": reward,
        }
        self._last_energy = avg_power_kw
        return reward, terms

    def _estimate_energy_usage(self) -> Tuple[float, float, float, int]:
        total_power = 0.0
        total_energy_kwh = 0.0
        cav_count = 0
        step_hours = self.config.step_length / 3600.0
        for lane_id in self.config.controlled_lanes:
            for vid in self._conn.lane.getLastStepVehicleIDs(lane_id):
                if not self._is_cav(vid):
                    continue
                cav_count += 1
                try:
                    v = max(self._conn.vehicle.getSpeed(vid), 0.0)
                except traci.TraCIException:
                    v = 0.0
                try:
                    a = self._conn.vehicle.getAcceleration(vid)
                except traci.TraCIException:
                    a = 0.0
                power_kw = self._longitudinal_power(v, a)
                total_power += power_kw
                total_energy_kwh += power_kw * step_hours
        if cav_count == 0:
            return 0.0, 0.0, 0.0, 0
        avg_power = total_power / cav_count
        avg_energy = total_energy_kwh / cav_count
        return avg_power, avg_energy, total_energy_kwh, cav_count

    @staticmethod
    def _longitudinal_power(speed: float, acceleration: float) -> float:
        mass = 1500.0
        c_rr = 0.015
        rho = 1.225
        drag_coeff = 0.32
        frontal_area = 2.2
        g = 9.81
        rolling = mass * g * c_rr * speed
        aero = 0.5 * rho * drag_coeff * frontal_area * speed ** 3
        inertial = mass * speed * acceleration
        power = rolling + aero + max(inertial, 0.0)
        return max(power, 0.0) * 0.001  # kW

    def _total_halting(self) -> Tuple[int, int]:
        cav_total = 0
        all_total = 0
        for lane_id in self.config.controlled_lanes:
            cav_lane, all_lane = self._count_halting(self._conn.lane.getLastStepVehicleIDs(lane_id))
            cav_total += cav_lane
            all_total += all_lane
        return cav_total, all_total

    def _count_halting(self, vehicles: Iterable[str]) -> Tuple[int, int]:
        cav = 0
        total = 0
        for vid in vehicles:
            try:
                speed = self._conn.vehicle.getSpeed(vid)
            except traci.TraCIException:
                speed = 0.0
            if speed < 0.1:
                total += 1
                if self._is_cav(vid):
                    cav += 1
        return cav, total

    def _total_waiting_time(self) -> Tuple[float, float]:
        cav_wait = 0.0
        total_wait = 0.0
        for lane_id in self.config.controlled_lanes:
            for vid in self._conn.lane.getLastStepVehicleIDs(lane_id):
                try:
                    wait = self._conn.vehicle.getWaitingTime(vid)
                except traci.TraCIException:
                    wait = 0.0
                total_wait += wait
                if self._is_cav(vid):
                    cav_wait += wait
        return cav_wait, total_wait

    def _is_cav(self, vehicle_id: str) -> bool:
        try:
            vtype = self._conn.vehicle.getTypeID(vehicle_id)
        except traci.TraCIException:
            return False
        return vtype in self.config.cav_type_ids

    # --------------------------- serialization -----------------------------

    def as_dict(self) -> Dict[str, float | str | Sequence[str]]:
        return asdict(self.config)


__all__ = ["CavSpeedEnv"]
