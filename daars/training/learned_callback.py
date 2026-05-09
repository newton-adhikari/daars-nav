"""
Callback for training with learned reward modulation.
"""

import sys
import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from daars.rewards.learned_daars import LearnedModulatorManager


class LearnedDAARSCallback(BaseCallback):
    """SB3 callback that manages the learned reward modulator."""

    def __init__(
        self,
        modulator: LearnedModulatorManager,
        reward_type: str = "learned",
        total_timesteps: int = 300_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.modulator = modulator
        self.reward_type = reward_type
        self.total_timesteps = total_timesteps

        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.successes: list[bool] = []
        self.collisions: list[bool] = []
        self.min_clearances: list[float] = []

        self._ep_reward = 0.0
        self._ep_len = 0
        self._ep_cost = 0.0
        self._start_time = None
        self._last_log = 0
        self._log_interval = 10_000

    def _on_training_start(self):
        self._start_time = time.time()

    def _on_step(self) -> bool:
        self._ep_reward += float(self.locals["rewards"][0])
        self._ep_len += 1

        # Track cost for modulator update
        infos = self.locals.get("infos", [{}])
        for info in infos:
            self._ep_cost += float(info.get("cost", 0.0))

        # Progress logging
        if self.num_timesteps - self._last_log >= self._log_interval:
            self._last_log = self.num_timesteps
            elapsed = time.time() - (self._start_time or time.time())
            fps = self.num_timesteps / max(elapsed, 1e-6)
            pct = 100.0 * self.num_timesteps / self.total_timesteps
            rem = (self.total_timesteps - self.num_timesteps) / max(fps, 1)
            n = len(self.successes)
            sr = float(np.mean(self.successes[-50:])) if n >= 50 else (float(np.mean(self.successes)) if n else 0)
            cr = float(np.mean(self.collisions[-50:])) if n >= 50 else (float(np.mean(self.collisions)) if n else 0)
            sys.stdout.write(
                f"\r  [{self.reward_type:10s}] {pct:5.1f}%  "
                f"{fps:6.0f} fps  ~{rem:5.0f}s  "
                f"ep={n:4d}  SR={sr:.3f}  CR={cr:.3f}    "
            )
            sys.stdout.flush()

        # Episode boundaries
        dones = self.locals.get("dones", [False])
        for done, info in zip(dones, infos):
            if not done:
                continue
            success = bool(info.get("reached_goal", False))
            collision = bool(info.get("collision", False))
            timeout = not success and not collision

            self.episode_rewards.append(self._ep_reward)
            self.episode_lengths.append(self._ep_len)
            self.successes.append(success)
            self.collisions.append(collision)
            self.min_clearances.append(float(info.get("min_clearance", float("inf"))))

            # Update modulator
            self.modulator.end_episode(
                cost=self._ep_cost / max(self._ep_len, 1),
                success=success,
                timeout=timeout,
            )

            self._ep_reward = 0.0
            self._ep_len = 0
            self._ep_cost = 0.0

        return True

    def get_curves(self) -> dict:
        return {
            "rewards": self.episode_rewards,
            "lengths": self.episode_lengths,
            "successes": self.successes,
            "collisions": self.collisions,
            "min_clearances": self.min_clearances,
        }