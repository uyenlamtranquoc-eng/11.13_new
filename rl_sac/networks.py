from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], output_dim: int, activation: type[nn.Module] = nn.ReLU) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(activation())
            prev = size
        layers.append(nn.Linear(prev, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        self.net = MLP(obs_dim + act_dim, hidden_sizes, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int], log_std_bounds: tuple[float, float] = (-5.0, 2.0)) -> None:
        super().__init__()
        self.base = MLP(obs_dim, hidden_sizes, 2 * act_dim)
        self.act_dim = act_dim
        self.log_std_bounds = log_std_bounds

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        mu_logstd = self.base(obs)
        mu, log_std = mu_logstd.chunk(2, dim=-1)
        low, high = self.log_std_bounds
        log_std = torch.tanh(log_std)
        log_std = low + 0.5 * (log_std + 1.0) * (high - low)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        return y_t, log_prob.sum(dim=-1, keepdim=True)

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mu, _ = self.forward(obs)
        return torch.tanh(mu)


__all__ = ["MLP", "QNetwork", "GaussianPolicy"]
