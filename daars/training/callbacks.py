"""SB3 callbacks for logging navigation metrics during training."""

from __future__ import annotations

import sys
import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class NavCallback(BaseCallback):
    """
    Logs navigation-specific episode metrics with progress in console itself.
    """

    def __init__(
        self,
        reward_type: str = "daars",
        total_timesteps: int = 1_000_000,
        log_interval: int = 10_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        
        self.reward_type     = reward_type
        self.total_timesteps = total_timesteps
        self.log_interval    = log_interval

        self.rewards         : list[float] = []
        self.lengths         : list[int]   = []
        self.successes       : list[bool]  = []
        self.collisions      : list[bool]  = []
        self.min_clearances  : list[float] = []
        self.cumulative_costs: list[float] = []

        self._ep_reward  = 0.0
        self._ep_len     = 0
        self._start_time = None
        self._last_log   = 0

    def _on_training_start(self):
        self._start_time = time.time()

    def _on_step(self) -> bool:
        self._ep_reward += float(self.locals["rewards"][0])
        self._ep_len    += 1

        # Live console progress
        if self.num_timesteps - self._last_log >= self.log_interval:
            self._last_log = self.num_timesteps
            elapsed = time.time() - (self._start_time or time.time())
            fps     = self.num_timesteps / max(elapsed, 1e-6)
            pct     = 100.0 * self.num_timesteps / self.total_timesteps
            rem     = (self.total_timesteps - self.num_timesteps) / max(fps, 1)
            n       = len(self.successes)
            sr = np.mean(self.successes[-50:])  if n >= 50 else (np.mean(self.successes) if n else 0)
            cr = np.mean(self.collisions[-50:]) if n >= 50 else (np.mean(self.collisions) if n else 0)
            sys.stdout.write(
                f"\r  [{self.reward_type:10s}] {pct:5.1f}%  "
                f"{fps:6.0f} fps  ~{rem:5.0f}s  "
                f"ep={n:4d}  SR={sr:.3f}  CR={cr:.3f}    "
            )
            sys.stdout.flush()

        # Episode boundaries
        for done, info in zip(
            self.locals.get("dones", []),
            self.locals.get("infos", []),
        ):
            if not done:
                continue
            self.rewards.append(self._ep_reward)
            self.lengths.append(self._ep_len)
            self.successes.append(bool(info.get("reached_goal", False)))
            self.collisions.append(bool(info.get("collision", False)))
            self.min_clearances.append(float(info.get("min_clearance", float("inf"))))
            self.cumulative_costs.append(float(info.get("cumulative_cost", 0.0)))
            self._ep_reward = 0.0
            self._ep_len    = 0

        return True

    def get_curves(self) -> dict:
        return {
            "rewards":          self.rewards,
            "lengths":          self.lengths,
            "successes":        self.successes,
            "collisions":       self.collisions,
            "min_clearances":   self.min_clearances,
            "cumulative_costs": self.cumulative_costs,
        }

