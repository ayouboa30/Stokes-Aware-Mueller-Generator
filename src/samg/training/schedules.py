"""Explicit phase schedule for SAMG and the direct VAE baseline."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Ramp:
    start: int
    length: int
    weight: float

    def at(self, epoch: int) -> float:
        if epoch < self.start:
            return 0.0
        if self.length <= 0:
            return float(self.weight)
        progress = min(1.0, (epoch - self.start) / self.length)
        return float(self.weight) * progress


@dataclass
class PenaltySchedule:
    """Ramp penalties and optionally draw one independent multiplier per epoch.

    Random multipliers affect regularizers only. The reconstruction objective is
    never randomized. Draws are reproducible for a fixed seed and are retained
    in the history file.
    """

    ramps: dict[str, Ramp]
    switch_epoch: int
    randomize: bool = False
    normalized: tuple[str, ...] = ("cond_in", "cond_out", "poincare")
    calibration_window: int = 20
    multipliers: dict[str, float] = field(default_factory=dict)
    references: dict[str, float] = field(default_factory=dict)
    observations: dict[str, list[float]] = field(default_factory=dict)

    def resample(self, generator: random.Random) -> dict[str, float]:
        self.multipliers = {
            name: generator.random() if self.randomize else 1.0 for name in self.ramps
        }
        return dict(self.multipliers)

    def weights(self, epoch: int) -> dict[str, float]:
        return {
            name: ramp.at(epoch) * self.multipliers.get(name, 1.0) / self.references.get(name, 1.0)
            for name, ramp in self.ramps.items()
        }

    def pending(self, epoch: int) -> tuple[str, ...]:
        return tuple(
            name
            for name, ramp in self.ramps.items()
            if name in self.normalized
            and name not in self.references
            and epoch >= max(1, ramp.start - self.calibration_window)
        )

    def observe(self, epoch: int, medians: dict[str, float]) -> None:
        """Calibrate unbounded physical terms before their ramp begins."""
        for name, ramp in self.ramps.items():
            if name not in self.normalized or name in self.references:
                continue
            if epoch >= max(1, ramp.start - self.calibration_window):
                self.observations.setdefault(name, []).append(abs(float(medians.get(name, 0.0))))
            if epoch >= max(1, ramp.start - 1):
                values = self.observations.get(name, [])
                reference = float(np.median(values)) if values else 0.0
                self.references[name] = reference if reference > 1e-12 else 1.0


def default_schedule(
    model: str,
    *,
    reconstruction_epochs: int = 200,
    switch_epoch: int = 300,
    ramp_length: int = 100,
    kl_weight: float = 1e-3,
) -> PenaltySchedule:
    if model not in {"samg", "samg-random", "operator", "direct"}:
        raise ValueError(f"Unknown model {model!r}")
    ramps = {"kl": Ramp(reconstruction_epochs, ramp_length, kl_weight)}
    if model != "direct":
        ramps.update(
            {
                "cond_in": Ramp(switch_epoch, ramp_length, 0.10),
                "cond_out": Ramp(switch_epoch, ramp_length, 0.10),
                "poincare": Ramp(switch_epoch, ramp_length, 0.05),
            }
        )
    return PenaltySchedule(
        ramps=ramps,
        switch_epoch=switch_epoch,
        randomize=model == "samg-random",
    )
