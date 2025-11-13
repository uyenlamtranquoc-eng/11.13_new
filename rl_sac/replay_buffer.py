from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class TransitionBatch:
    obs: torch.Tensor
    act: torch.Tensor
    rew: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        capacity: int,
        num_phase_bins: int | None = None,
        balance_by_phase: bool = False,
    ) -> None:
        self.capacity = capacity
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.done_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.phase_ids = np.full(capacity, -1, dtype=np.int16)
        self.ptr = 0
        self.size = 0
        self.num_phase_bins = num_phase_bins if balance_by_phase and num_phase_bins else None
        self.balance_by_phase = balance_by_phase and self.num_phase_bins is not None

    def push(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
        phase_id: int | None = None,
    ) -> None:
        idx = self.ptr
        self.obs_buf[idx] = obs
        self.next_obs_buf[idx] = next_obs
        self.act_buf[idx] = act
        self.rew_buf[idx] = rew
        self.done_buf[idx] = float(done)
        if self.balance_by_phase:
            if phase_id is None:
                phase_idx = -1
            else:
                phase_idx = int(phase_id) % max(self.num_phase_bins or 1, 1)
            self.phase_ids[idx] = phase_idx
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> TransitionBatch:
        if self.size < batch_size:
            raise ValueError("Not enough samples in buffer")
        idxs = self._sample_indices(batch_size)
        obs = torch.as_tensor(self.obs_buf[idxs], device=device)
        act = torch.as_tensor(self.act_buf[idxs], device=device)
        rew = torch.as_tensor(self.rew_buf[idxs], device=device)
        next_obs = torch.as_tensor(self.next_obs_buf[idxs], device=device)
        done = torch.as_tensor(self.done_buf[idxs], device=device)
        return TransitionBatch(obs, act, rew, next_obs, done)

    def _sample_indices(self, batch_size: int) -> np.ndarray:
        if not self.balance_by_phase or self.num_phase_bins is None:
            return np.random.randint(0, self.size, size=batch_size)
        per_bin = max(batch_size // self.num_phase_bins, 1)
        available = self.phase_ids[: self.size]
        collected: list[np.ndarray] = []
        for phase in range(self.num_phase_bins):
            bin_idxs = np.where(available == phase)[0]
            if len(bin_idxs) == 0:
                continue
            take = min(per_bin, len(bin_idxs))
            chosen = np.random.choice(bin_idxs, size=take, replace=False)
            collected.append(chosen)
        if not collected:
            return np.random.randint(0, self.size, size=batch_size)
        idxs = np.concatenate(collected)
        if len(idxs) < batch_size:
            remainder = np.random.randint(0, self.size, size=batch_size - len(idxs))
            idxs = np.concatenate([idxs, remainder])
        return idxs[:batch_size]

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "obs": self.obs_buf,
            "act": self.act_buf,
            "rew": self.rew_buf,
            "next_obs": self.next_obs_buf,
            "done": self.done_buf,
            "ptr": np.array([self.ptr]),
            "size": np.array([self.size]),
            "phase_ids": self.phase_ids,
        }

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        self.obs_buf = state["obs"]
        self.act_buf = state["act"]
        self.rew_buf = state["rew"]
        self.next_obs_buf = state["next_obs"]
        self.done_buf = state["done"]
        self.ptr = int(state["ptr"][0])
        self.size = int(state["size"][0])
        if "phase_ids" in state:
            self.phase_ids = state["phase_ids"]


__all__ = ["ReplayBuffer", "TransitionBatch"]
