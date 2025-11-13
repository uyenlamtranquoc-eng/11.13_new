from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from rl_sac.config import TrainConfig
from rl_sac.sac_agent import SACAgent
from rl_sac.sumo_env import CavSpeedEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SAC controller")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--gui", action="store_true", help="Enable SUMO GUI")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig()
    env = CavSpeedEnv(config.env, use_gui=args.gui)
    obs_dim = env.observation_dim
    act_dim = env.action_dim
    device = torch.device(args.device) if args.device else None
    agent = SACAgent(obs_dim, act_dim, config.sac, device=device)
    agent.load(Path(args.checkpoint))

    rewards = []
    energies = []
    queues = []
    for ep in range(args.episodes):
        obs = env.reset(seed=config.env.seed + 10_000 + ep)
        done = False
        ep_reward = 0.0
        ep_energy = 0.0
        ep_queue = 0.0
        steps = 0
        while not done:
            action = agent.select_action(obs, deterministic=True).numpy()
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_energy += info.get("energy", 0.0)
            ep_queue += info.get("queue", 0.0)
            steps += 1
        rewards.append(ep_reward)
        energies.append(ep_energy / max(steps, 1))
        queues.append(ep_queue / max(steps, 1))
        print(f"Episode {ep+1}: reward={ep_reward:.2f}, energy={energies[-1]:.3f}, queue={queues[-1]:.2f}")

    print("Summary -> reward: {:.2f}¡À{:.2f}, energy: {:.3f}, queue: {:.2f}".format(
        np.mean(rewards), np.std(rewards), np.mean(energies), np.mean(queues)
    ))
    env.close()


if __name__ == "__main__":
    main()
