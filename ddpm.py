from __future__ import annotations

import math
import random
from typing import Callable, Optional


class DDPM:
    """Simple scalar DDPM implementation."""

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
    ) -> None:
        if timesteps < 1:
            raise ValueError("timesteps must be >= 1")
        if not (0.0 < beta_start < 1.0 and 0.0 < beta_end < 1.0):
            raise ValueError("beta_start and beta_end must be in (0, 1)")

        self.timesteps = timesteps
        self.betas = self._build_betas(
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule=schedule,
        )
        self.alphas = [1.0 - b for b in self.betas]

        self.alpha_bars = []
        running = 1.0
        for a in self.alphas:
            running *= a
            self.alpha_bars.append(running)

    @staticmethod
    def _build_betas(
        timesteps: int,
        beta_start: float,
        beta_end: float,
        schedule: str,
    ) -> list[float]:
        if schedule == "linear":
            return [
                beta_start + (beta_end - beta_start) * (i / max(timesteps - 1, 1))
                for i in range(timesteps)
            ]
        if schedule == "cosine":
            return DDPM._cosine_betas(timesteps)
        raise ValueError(
            f"Unsupported schedule '{schedule}'. Use 'linear' or 'cosine'."
        )

    @staticmethod
    def _cosine_betas(timesteps: int, s: float = 0.008) -> list[float]:
        def alpha_bar_fn(step: int) -> float:
            t = step / timesteps
            angle = ((t + s) / (1.0 + s)) * (math.pi / 2.0)
            return math.cos(angle) ** 2

        alpha_bar_values = [
            alpha_bar_fn(step) / alpha_bar_fn(0) for step in range(timesteps + 1)
        ]
        return [
            min(1.0 - (alpha_bar_values[i + 1] / alpha_bar_values[i]), 0.999)
            for i in range(timesteps)
        ]

    def q_sample(self, x0: float, t: int, noise: Optional[float] = None) -> float:
        """Forward diffusion: q(x_t | x_0)."""
        self._check_timestep(t)
        if noise is None:
            noise = random.gauss(0.0, 1.0)

        alpha_bar_t = self.alpha_bars[t]
        return math.sqrt(alpha_bar_t) * x0 + math.sqrt(1.0 - alpha_bar_t) * noise

    def predict_x0(self, xt: float, t: int, pred_noise: float) -> float:
        """Estimate x_0 from x_t and predicted noise."""
        self._check_timestep(t)
        alpha_bar_t = self.alpha_bars[t]
        return (xt - math.sqrt(1.0 - alpha_bar_t) * pred_noise) / math.sqrt(alpha_bar_t)

    def p_sample(self, xt: float, t: int, pred_noise: float) -> float:
        """Single reverse step: p(x_{t-1} | x_t)."""
        self._check_timestep(t)
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        beta_t = self.betas[t]

        mean = (1.0 / math.sqrt(alpha_t)) * (
            xt - ((1.0 - alpha_t) / math.sqrt(1.0 - alpha_bar_t)) * pred_noise
        )

        if t == 0:
            return mean
        return mean + math.sqrt(beta_t) * random.gauss(0.0, 1.0)

    def sample(
        self, model: Callable[[float, int], float], xT: Optional[float] = None
    ) -> float:
        """Generate one scalar sample using a noise-predicting model."""
        xt = random.gauss(0.0, 1.0) if xT is None else xT
        for t in range(self.timesteps - 1, -1, -1):
            pred_noise = model(xt, t)
            xt = self.p_sample(xt, t, pred_noise)
        return xt

    def _check_timestep(self, t: int) -> None:
        if not (0 <= t < self.timesteps):
            raise ValueError(f"t must be in [0, {self.timesteps - 1}], got {t}")
