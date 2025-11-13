from __future__ import annotations

import os
import random
import subprocess
import sys
from collections import deque
from dataclasses import asdict
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
        # phase one-hot (phase_bins) and phase progress (1)
        return per_lane + 5 + self.config.phase_bins + 1

    @property
    def observation_dim(self) -> int:
        return self._frame_dim * self.config.observation_history

    # ------------------------------ public api -----------------------------

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.close()
        if seed is not None:
            self._rng.seed(seed)
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
        phase_one_hot, phase_progress, phase_idx = self._phase_features()
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
            halting = self._count_halting(vehicles)
            queue_veh += halting
            capacity = self._lane_capacities.get(lane_id, max(len_lane / max(cfg.density_headway, 1.0), 1.0))
            density = min(len(vehicles) / max(capacity, 1.0), 1.0)
            cav_share = cav_count / max(len(vehicles), 1)
            mean_speed = (sum(speeds) / len(speeds)) if speeds else 0.0
            wait_avg = (sum(waits) / len(waits)) if waits else 0.0
            halting_ratio = halting / max(len(vehicles), 1)
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
            + [phase_progress]
        )
        return np.array(features, dtype=np.float32)

    def _current_phase_index(self) -> int:
        if self._conn is None:
            return 0
        try:
            return int(self._conn.trafficlight.getPhase(self.config.phase_signal_id))
        except traci.TraCIException:
            return 0

    def _phase_features(self) -> Tuple[np.ndarray, float, int]:
        cfg = self.config
        phase_idx = self._current_phase_index()
        one_hot = np.zeros(cfg.phase_bins, dtype=np.float32)
        one_hot[min(phase_idx, cfg.phase_bins - 1)] = 1.0
        progress = 0.0
        if self._conn is not None:
            try:
                duration = max(self._conn.trafficlight.getPhaseDuration(cfg.phase_signal_id), cfg.step_length)
                next_switch = self._conn.trafficlight.getNextSwitch(cfg.phase_signal_id)
                sim_time = self._conn.simulation.getTime()
                remaining = max(next_switch - sim_time, 0.0)
                progress = 1.0 - min(remaining / duration, 1.0)
            except traci.TraCIException:
                progress = 0.0
        return one_hot, progress, phase_idx

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

    def _apply_speed_commands(self, lane_targets: np.ndarray) -> np.ndarray:
        cfg = self.config
        applied = np.zeros_like(lane_targets)
        self._prev_speed_targets = self._lane_target_speeds.copy()
        phase_idx = self._current_phase_index()
        if phase_idx in cfg.mainline_green_phases:
            phase_scale = cfg.phase_speed_scale_green
        elif phase_idx in cfg.mainline_yellow_phases:
            phase_scale = cfg.phase_speed_scale_yellow
        else:
            phase_scale = cfg.phase_speed_scale_red
        phase_scale = float(np.clip(phase_scale, 0.0, 1.0))
        phase_high = cfg.action_low + (cfg.action_high - cfg.action_low) * phase_scale
        phase_high = max(cfg.action_low, phase_high)
        for lane_idx, lane_id in enumerate(cfg.controlled_lanes):
            desired = float(np.clip(lane_targets[lane_idx], cfg.action_low, phase_high))
            last = self._lane_target_speeds[lane_idx]
            max_delta = max(cfg.max_speed_delta, 1e-3)
            limited = float(np.clip(desired, last - max_delta, last + max_delta))
            self._lane_target_speeds[lane_idx] = limited
            applied[lane_idx] = limited
            vehicles = self._conn.lane.getLastStepVehicleIDs(lane_id)
            for vid in vehicles:
                if not self._is_cav(vid):
                    continue
                if self._conn.vehicle.getLaneIndex(vid) != 1:
                    # Ensure we only touch the straight-through lane.
                    continue
                self._conn.vehicle.setSpeedMode(vid, 0b00000)
                self._conn.vehicle.setSpeed(vid, limited)
        return applied

    # ---------------------------- reward logic -----------------------------

    def _compute_reward(self, frame: np.ndarray) -> Tuple[float, Dict[str, float]]:
        cfg = self.config
        weights = cfg.reward_weights
        energy = self._estimate_energy_usage()
        halting = self._total_halting()
        wait = self._total_waiting_time()
        arrived_total = self._conn.simulation.getArrivedNumber()
        arrived_delta = max(arrived_total - self._last_arrived, 0)
        self._last_arrived = arrived_total
        queue_ratio = min(halting / max(self._max_queue_capacity, 1.0), 1.0)
        energy_norm = min(energy / max(cfg.energy_normalizer, 1e-3), 1.0)
        wait_norm = min(wait / max(cfg.corridor_wait_normalizer, 1.0), 1.0)
        throughput_norm = min(arrived_delta / max(cfg.throughput_normalizer, 1.0), 1.0)
        speed_delta = (self._lane_target_speeds - self._prev_speed_targets) / max(cfg.max_speed_delta, 1e-3)
        smooth = float(np.mean(np.square(speed_delta)))
        phase_idx = self._current_phase_index()
        is_green = phase_idx in cfg.mainline_green_phases
        is_yellow = phase_idx in cfg.mainline_yellow_phases
        is_red = not (is_green or is_yellow)
        red_queue_ratio = queue_ratio if is_red else 0.0
        red_wait_norm = wait_norm if is_red else 0.0
        green_throughput_norm = throughput_norm if is_green else 0.0

        reward = (
            -weights.energy * energy_norm
            -weights.queue * queue_ratio
            -weights.wait * wait_norm
            -weights.smooth * smooth
            +weights.throughput * throughput_norm
            +weights.green_phase_bonus * green_throughput_norm
            -weights.red_phase_wait * red_wait_norm
            -weights.red_phase_wait * red_queue_ratio
        )
        terms = {
            "energy": energy,
            "energy_norm": energy_norm,
            "queue": float(halting),
            "queue_ratio": float(queue_ratio),
            "wait_time": float(wait),
            "wait_norm": wait_norm,
            "smooth": smooth,
            "throughput": float(arrived_delta),
            "throughput_norm": throughput_norm,
            "red_queue_ratio": float(red_queue_ratio),
            "red_wait_norm": red_wait_norm,
            "green_throughput_norm": green_throughput_norm,
            "reward": reward,
        }
        self._last_energy = energy
        return reward, terms

    def _estimate_energy_usage(self) -> float:
        total_energy = 0.0
        cav_count = 0
        for lane_id in self.config.controlled_lanes:
            for vid in self._conn.lane.getLastStepVehicleIDs(lane_id):
                if not self._is_cav(vid):
                    continue
                cav_count += 1
                v = max(self._conn.vehicle.getSpeed(vid), 0.0)
                a = self._conn.vehicle.getAcceleration(vid)
                total_energy += self._longitudinal_power(v, a)
        if cav_count == 0:
            return 0.0
        return total_energy / cav_count

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
        return max(power, 0.0) * 0.001  # scale kW for reward stability

    def _total_halting(self) -> int:
        halt = 0
        for lane_id in self.config.controlled_lanes:
            halt += self._count_halting(self._conn.lane.getLastStepVehicleIDs(lane_id))
        return halt

    def _count_halting(self, vehicles: Iterable[str]) -> int:
        total = 0
        for vid in vehicles:
            if self._conn.vehicle.getSpeed(vid) < 0.1:
                total += 1
        return total

    def _total_waiting_time(self) -> float:
        wait = 0.0
        for lane_id in self.config.controlled_lanes:
            for vid in self._conn.lane.getLastStepVehicleIDs(lane_id):
                if self._is_cav(vid):
                    wait += self._conn.vehicle.getWaitingTime(vid)
        return wait

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
