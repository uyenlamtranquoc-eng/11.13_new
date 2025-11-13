from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from dataclasses import replace

from rl_sac.config import TrainConfig
from rl_sac.replay_buffer import ReplayBuffer
from rl_sac.sac_agent import SACAgent
from rl_sac.sumo_env import CavSpeedEnv
from rl_sac.utils.logger import EpisodeLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC speed controller for SUMO corridor")
    parser.add_argument("--episodes", type=int, help="Total training episodes", default=None)
    parser.add_argument("--gui", action="store_true", help="Visualize SUMO GUI during training")
    parser.add_argument("--config", type=str, help="Optional JSON config override", default=None)
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda)")
    return parser.parse_args()


def maybe_override_config(base: TrainConfig, override_path: str | None) -> TrainConfig:
    if not override_path:
        return base
    path = Path(override_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    # simple shallow overrides
    for field, value in data.items():
        if hasattr(base, field):
            attr = getattr(base, field)
            if isinstance(value, dict) and hasattr(attr, "__dict__"):
                for sub_key, sub_val in value.items():
                    if hasattr(attr, sub_key):
                        setattr(attr, sub_key, sub_val)
            else:
                setattr(base, field, value)
    return base


def evaluate(agent: SACAgent, env: CavSpeedEnv, episodes: int = 3) -> Dict[str, float]:
    metrics: Dict[str, float] = {"reward": 0.0}
    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.select_action(obs, deterministic=True).numpy()
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
        metrics["reward"] += ep_reward
    metrics = {k: v / episodes for k, v in metrics.items()}
    return metrics


def train() -> None:
    args = parse_args()
    config = maybe_override_config(TrainConfig(), args.config)
    if args.episodes:
        config.total_episodes = args.episodes
    env = CavSpeedEnv(config.env, use_gui=args.gui)
    eval_env = CavSpeedEnv(replace(config.env, port=config.env.port + 1), use_gui=False)
    obs_dim = env.observation_dim
    act_dim = env.action_dim
    device = torch.device(args.device) if args.device else None
    agent = SACAgent(obs_dim, act_dim, config.sac, device=device)
    buffer = ReplayBuffer(
        obs_dim,
        act_dim,
        config.sac.replay_capacity,
        num_phase_bins=config.env.phase_bins,
        balance_by_phase=config.sac.phase_balanced_replay,
    )
    metrics_dir = config.run_dir / "train"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logger = EpisodeLogger(metrics_dir)
    metrics_csv = metrics_dir / "metrics.csv"
    csv_file = metrics_csv.open("a", newline="", encoding="utf-8")
    fieldnames = [
        "episode",
        "global_step",
        "episode_reward",
        "running_avg_reward",
        "eval_reward",
        "mean_energy",
        "mean_energy_norm",
        "mean_queue",
        "mean_queue_ratio",
        "mean_wait_norm",
        "mean_throughput_norm",
        "mean_red_queue_ratio",
        "q1_loss",
        "actor_loss",
        "alpha",
    ]
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if metrics_csv.stat().st_size == 0:
        csv_writer.writeheader()

    global_step = 0
    best_eval = -float("inf")
    last_eval_reward = 0.0
    try:
        for episode in range(1, config.total_episodes + 1):
            obs = env.reset(seed=config.env.seed + episode)
            episode_reward = 0.0
            info: Dict[str, float] = {}
            energy_norm_sum = 0.0
            energy_raw_sum = 0.0
            queue_ratio_sum = 0.0
            queue_raw_sum = 0.0
            throughput_norm_sum = 0.0
            wait_norm_sum = 0.0
            red_queue_ratio_sum = 0.0
            step_count = 0
            for t in range(config.max_episode_steps):
                if global_step < config.sac.start_random_steps:
                    action = np.random.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
                else:
                    action = agent.select_action(obs, deterministic=False).numpy()
                next_obs, reward, done, info = env.step(action)
                buffer.push(
                    obs,
                    action,
                    reward,
                    next_obs,
                    done,
                    phase_id=int(info.get("phase_bin", -1)),
                )
                obs = next_obs
                episode_reward += reward
                global_step += 1
                step_count += 1
                energy_norm_sum += info.get("energy_norm", 0.0)
                energy_raw_sum += info.get("energy", 0.0)
                queue_ratio_sum += info.get("queue_ratio", 0.0)
                queue_raw_sum += info.get("queue", 0.0)
                throughput_norm_sum += info.get("throughput_norm", 0.0)
                wait_norm_sum += info.get("wait_norm", 0.0)
                red_queue_ratio_sum += info.get("red_queue_ratio", 0.0)

                if global_step >= config.sac.update_after and global_step % config.sac.update_every == 0:
                    stats = agent.update(buffer)
                    for key, value in stats.items():
                        logger.log(key, value)

                if done:
                    break

            mean_energy = 0.0
            mean_energy_norm = 0.0
            mean_queue = 0.0
            mean_queue_ratio = 0.0
            mean_throughput_norm = 0.0
            mean_wait_norm = 0.0
            mean_red_queue_ratio = 0.0
            if step_count:
                inv_steps = 1.0 / step_count
                mean_energy = energy_raw_sum * inv_steps
                mean_energy_norm = energy_norm_sum * inv_steps
                mean_queue = queue_raw_sum * inv_steps
                mean_queue_ratio = queue_ratio_sum * inv_steps
                mean_throughput_norm = throughput_norm_sum * inv_steps
                mean_wait_norm = wait_norm_sum * inv_steps
                mean_red_queue_ratio = red_queue_ratio_sum * inv_steps
                logger.log("energy", mean_energy)
                logger.log("energy_norm", mean_energy_norm)
                logger.log("queue", mean_queue)
                logger.log("queue_ratio", mean_queue_ratio)
                logger.log("throughput_norm", mean_throughput_norm)
                logger.log("wait_norm", mean_wait_norm)
                logger.log("red_queue_ratio", mean_red_queue_ratio)
            logger.log("episode_reward", episode_reward)
            if episode % config.eval_interval == 0:
                eval_stats = evaluate(agent, eval_env, episodes=2)
                last_eval_reward = eval_stats["reward"]
                logger.log("eval_reward", last_eval_reward)
                if last_eval_reward > best_eval:
                    best_eval = last_eval_reward
                    agent.save(config.model_dir / "best.pt")
            if episode % config.checkpoint_interval == 0:
                agent.save(config.model_dir / f"checkpoint_{episode}.pt")

            summary = logger.summary()
            csv_writer.writerow(
                {
                    "episode": episode,
                    "global_step": global_step,
                    "episode_reward": episode_reward,
                    "running_avg_reward": summary.get("episode_reward", 0.0),
                    "eval_reward": last_eval_reward,
                    "mean_energy": mean_energy,
                    "mean_energy_norm": mean_energy_norm,
                    "mean_queue": mean_queue,
                    "mean_queue_ratio": mean_queue_ratio,
                    "mean_wait_norm": mean_wait_norm,
                    "mean_throughput_norm": mean_throughput_norm,
                    "mean_red_queue_ratio": mean_red_queue_ratio,
                    "q1_loss": summary.get("q1_loss", 0.0),
                    "actor_loss": summary.get("actor_loss", 0.0),
                    "alpha": summary.get("alpha", 0.0),
                }
            )
            csv_file.flush()

            if episode % config.print_interval == 0:
                print(
                    "Episode {}/{} | step_reward={:.3f} | avg_reward={:.3f} | eval={:.3f} | "
                    "energy_norm={:.3f} | queue_ratio={:.3f} | q1={:.4f} | actor={:.4f} | alpha={:.4f}".format(
                        episode,
                        config.total_episodes,
                        episode_reward,
                        summary.get("episode_reward", 0.0),
                        summary.get("eval_reward", 0.0),
                        summary.get("energy_norm", 0.0),
                        summary.get("queue_ratio", 0.0),
                        summary.get("q1_loss", 0.0),
                        summary.get("actor_loss", 0.0),
                        summary.get("alpha", 0.0),
                    )
                )
    finally:
        csv_file.close()
        env.close()
        eval_env.close()


if __name__ == "__main__":
    train()
