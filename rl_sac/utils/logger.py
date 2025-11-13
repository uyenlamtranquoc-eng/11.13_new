from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class MetricHistory:
    name: str
    values: List[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        self.values.append(value)

    def mean(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)


class EpisodeLogger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.scalars: Dict[str, MetricHistory] = {}
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def log(self, key: str, value: float) -> None:
        metric = self.scalars.setdefault(key, MetricHistory(key))
        metric.add(value)

    def summary(self) -> Dict[str, float]:
        return {key: metric.mean() for key, metric in self.scalars.items()}


__all__ = ["EpisodeLogger"]
