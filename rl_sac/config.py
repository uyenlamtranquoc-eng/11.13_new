from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence


@dataclass
class RewardWeights:
    energy: float = 0.7
    queue: float = 0.6
    wait: float = 0.45
    hdv_wait: float = 0.35
    smooth: float = 0.1
    throughput: float = 0.4
    green_wave: float = 0.25
    phase_penalty: float = 0.5


@dataclass
class EnvConfig:
    sumo_config: Path = Path(__file__).resolve().parent.parent / "sumo" / "grid1x3.sumocfg"
    sumo_binary: str = "sumo"
    sumo_gui_binary: str = "sumo-gui"
    host: str = "127.0.0.1"
    port: int = 8873
    step_length: float = 3.0
    max_steps: int = 1200
    warmup_steps: int = 120
    action_low: float = 0.0
    action_high: float = 15.0
    max_speed_delta: float = 2.0  # tighter delta encourages smoother decel profiles
    cav_speed_mode: int = 0b01101  # obey tls + speed limit + safe distance
    cav_type_ids: Sequence[str] = ("cav",)
    stopline_buffer: float = 30.0  # meters before stop line to clamp during red
    stopline_buffer_yellow: float = 20.0
    red_phase_speed_cap: float = 2.0
    yellow_phase_speed_cap: float = 6.0
    speed_limit_scale: float = 0.95
    controlled_lanes_e2w: Sequence[str] = (
        "E_J2_1",
        "J2_J1_1",
        "J1_J0_1",
        "J0_W_1",
    )
    controlled_lanes_w2e: Sequence[str] = (
        "W_J0_1",
        "J0_J1_1",
        "J1_J2_1",
        "J2_E_1",
    )
    observation_history: int = 8
    density_headway: float = 7.0  # meters, used to normalize lane density
    wait_normalizer: float = 300.0  # seconds
    corridor_cav_normalizer: float = 200.0
    corridor_flow_normalizer: float = 300.0
    corridor_wait_normalizer: float = 3000.0
    corridor_speed_normalizer: float = 15.0
    energy_normalizer: float = 80.0  # kW
    throughput_normalizer: float = 6.0  # veh per step
    phase_cycle: float = 170.0
    phase_bins: int = 8
    phase_balance_bins: int = 3
    phase_signal_id: str = "J1"
    mainline_green_phases: Sequence[int] = (0, 4)
    mainline_yellow_phases: Sequence[int] = (1, 3, 5, 7)
    phase_speed_scale_green: float = 1.0
    phase_speed_scale_yellow: float = 0.7
    phase_speed_scale_red: float = 0.25
    green_wave_window: float = 0.4
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    seed: int = 7
    sumo_output_root: Path | None = Path("artifacts/sumo_outputs")
    queue_output_dir: Path | None = None
    tripinfo_output_dir: Path | None = None
    fcd_output_dir: Path | None = None
    summary_output_dir: Path | None = None

    @property
    def controlled_lanes(self) -> List[str]:
        return list(self.controlled_lanes_e2w) + list(self.controlled_lanes_w2e)


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    hidden_units: Sequence[int] = (256, 256)
    replay_capacity: int = 200_000
    batch_size: int = 256
    target_entropy_scale: float = 0.98
    grad_steps_per_update: int = 1
    start_random_steps: int = 200
    update_after: int = 1000
    update_every: int = 1
    alpha_init: float = 0.2
    normalize_rewards: bool = True
    phase_balanced_replay: bool = True


@dataclass
class TrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    total_episodes: int = 120
    max_episode_steps: int = 1200
    eval_interval: int = 10
    model_dir: Path = Path("artifacts/models")
    run_dir: Path = Path("artifacts/runs")
    checkpoint_interval: int = 25
    deterministic_eval: bool = True
    print_interval: int = 1


__all__ = [
    "RewardWeights",
    "EnvConfig",
    "SACConfig",
    "TrainConfig",
]
