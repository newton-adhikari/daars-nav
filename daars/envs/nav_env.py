"""
2-D robot navigation environment.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RobotNavEnv(gym.Env):
    """2-D navigation with LiDAR, domain randomisation, and cost signals."""

    def __init__(
        self,
        config: dict,    ):
        super().__init__()
        self.cfg       = config["environment"]
        self.rcfg      = config["reward"]

        # Base env parameters
        self._base_arena   = float(self.cfg["arena_size"])
        

        # Spaces
        self.action_space = spaces.Box(
            low=np.array([-self._max_lin, -self._max_ang], dtype=np.float32),
            high=np.array([self._max_lin,  self._max_ang], dtype=np.float32),
        )
        obs_dim = self.num_rays + 4  # lidar + goal_dist + goal_angle + v_lin + v_ang
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # buffers
        self._base_angles  = np.linspace(0, 2 * np.pi, self.num_rays, endpoint=False)
        self._ranges_buf   = np.empty(self.num_rays, dtype=np.float64)
        self._obs_buf      = np.empty(obs_dim, dtype=np.float32)

        # Normalisation constants
        self._inv_lidar    = 1.0 / self.lidar_range
        self._inv_pi       = 1.0 / np.pi

        # Episode state (initialised in reset)
        self.robot_pos          = np.zeros(2)
        self.robot_theta        = 0.0
        self.robot_vel          = np.zeros(2)
        self.goal_pos           = np.zeros(2)
        self.arena              = self._base_arena
        self.step_count         = 0
        self.prev_goal_dist     = 0.0
        self.start_goal_dist    = 0.0   
        self.episode_min_clearance = float("inf")
        self.episode_path_length   = 0.0
        self.prev_pos           = np.zeros(2)
        self.episode_cost_sum   = 0.0


    # Gymnasium interface

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        

        margin = 1.5
        # Robot start
        self.robot_pos   = self.rng.uniform(margin, self.arena - margin, size=2)
        self.robot_theta = self.rng.uniform(-np.pi, np.pi)
        self.robot_vel[:] = 0.0

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

    