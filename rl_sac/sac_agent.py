from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim

from .config import SACConfig
from .networks import GaussianPolicy, QNetwork
from .replay_buffer import ReplayBuffer


class SACAgent:
    def __init__(self, obs_dim: int, act_dim: int, config: SACConfig, device: Optional[torch.device] = None) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = GaussianPolicy(obs_dim, act_dim, config.hidden_units).to(self.device)
        self.q1 = QNetwork(obs_dim, act_dim, config.hidden_units).to(self.device)
        self.q2 = QNetwork(obs_dim, act_dim, config.hidden_units).to(self.device)
        self.q1_target = QNetwork(obs_dim, act_dim, config.hidden_units).to(self.device)
        self.q2_target = QNetwork(obs_dim, act_dim, config.hidden_units).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=config.critic_lr)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=config.critic_lr)

        self.log_alpha = torch.tensor([config.alpha_init], requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.target_entropy = -config.target_entropy_scale * act_dim

        self.total_updates = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, obs: torch.Tensor | np.ndarray | None = None, deterministic: bool = False) -> torch.Tensor:
        if obs is None:
            raise ValueError("Observation is required to select an action.")
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(obs)
            else:
                action, _ = self.actor.sample(obs)
        return action.squeeze(0).cpu()

    def update(self, buffer: ReplayBuffer) -> Dict[str, float]:
        cfg = self.cfg
        stats: Dict[str, float] = {}
        for _ in range(cfg.grad_steps_per_update):
            batch = buffer.sample(cfg.batch_size, self.device)
            with torch.no_grad():
                next_action, next_log_prob = self.actor.sample(batch.next_obs)
                q1_target = self.q1_target(batch.next_obs, next_action)
                q2_target = self.q2_target(batch.next_obs, next_action)
                min_q = torch.min(q1_target, q2_target) - self.alpha * next_log_prob
                target_q = batch.rew + (1.0 - batch.done) * cfg.gamma * min_q

            q1_loss = F.mse_loss(self.q1(batch.obs, batch.act), target_q)
            q2_loss = F.mse_loss(self.q2(batch.obs, batch.act), target_q)
            self.q1_optim.zero_grad()
            q1_loss.backward()
            self.q1_optim.step()
            self.q2_optim.zero_grad()
            q2_loss.backward()
            self.q2_optim.step()

            new_action, log_prob = self.actor.sample(batch.obs)
            q_new = torch.min(self.q1(batch.obs, new_action), self.q2(batch.obs, new_action))
            actor_loss = (self.alpha * log_prob - q_new).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self._soft_update(self.q1_target, self.q1, cfg.tau)
            self._soft_update(self.q2_target, self.q2, cfg.tau)

            self.total_updates += 1
            stats = {
                "q1_loss": float(q1_loss.item()),
                "q2_loss": float(q2_loss.item()),
                "actor_loss": float(actor_loss.item()),
                "alpha_loss": float(alpha_loss.item()),
                "alpha": float(self.alpha.item()),
            }
        return stats

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "q1_target": self.q1_target.state_dict(),
                "q2_target": self.q2_target.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "config": self.cfg,
            },
            path,
        )

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(state["actor"])
        self.q1.load_state_dict(state["q1"])
        self.q2.load_state_dict(state["q2"])
        self.q1_target.load_state_dict(state["q1_target"])
        self.q2_target.load_state_dict(state["q2_target"])
        self.log_alpha = state["log_alpha"].clone().detach().to(self.device).requires_grad_(True)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.cfg.alpha_lr)

    @staticmethod
    def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


__all__ = ["SACAgent"]
