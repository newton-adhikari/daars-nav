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

