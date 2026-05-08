"""
2-D robot navigation environment.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .obstacles import ObstacleField


class RobotNavEnv(gym.Env):
    """2-D navigation with LiDAR, domain randomisation, and cost signals."""

    def __init__(
        self,
        config: dict,
        reward_fn=None,
        cost_fn=None,
        scenario: str = "simple",
        render_mode=None,
        seed: int | None = None,
        domain_rand: bool = True,   
    ):
        super().__init__()
        self.cfg       = config["environment"]
        self.rcfg      = config["reward"]
        self.reward_fn = reward_fn
        self.cost_fn   = cost_fn
        self.scenario  = scenario
        self.render_mode = render_mode
        self.domain_rand = domain_rand and self.cfg.get("domain_randomisation", {}).get("enable", False)
        self.rng = np.random.default_rng(seed)

        # Base env parameters
        self._base_arena   = float(self.cfg["arena_size"])
        self.dt            = float(self.cfg["dt"])
        self.max_steps     = int(self.cfg["max_steps"])
        self.robot_radius  = float(self.cfg["robot_radius"])
        self.goal_threshold = float(self.cfg["goal_threshold"])
        self.num_rays      = int(self.cfg["num_lidar_rays"])
        self.lidar_range   = float(self.cfg["lidar_max_range"])
        self._max_lin      = float(self.cfg["max_linear_vel"])
        self._max_ang      = float(self.cfg["max_angular_vel"])

        # Domain rand bounds
        dr = self.cfg.get("domain_randomisation", {})
        self._lidar_noise_std    = float(dr.get("lidar_noise_std", 0.0))
        self._lidar_dropout_rate = float(dr.get("lidar_dropout_rate", 0.0))
        self._vel_noise_std      = float(dr.get("vel_noise_std", 0.0))
        arena_range              = dr.get("arena_size_range", [self._base_arena] * 2)
        self._arena_range        = (float(arena_range[0]), float(arena_range[1]))

        # Dynamics domain randomisation this is for (sim-to-real transfer)
        self._action_delay_range = tuple(dr.get("action_delay_steps", [0, 3]))
        self._vel_smoothing_range = tuple(dr.get("vel_smoothing_alpha", [0.3, 1.0]))
        self._action_delay_steps = 0
        self._vel_smooth_alpha   = 1.0
        self._action_buffer      = []
        self._smooth_v_lin       = 0.0
        self._smooth_v_ang       = 0.0

        # Spaces
        self.action_space = spaces.Box(
            low=np.array([-self._max_lin, -self._max_ang], dtype=np.float32),
            high=np.array([self._max_lin,  self._max_ang], dtype=np.float32),
        )
        obs_dim = self.num_rays + 4  # lidar + goal_dist + goal_angle + v_lin + v_ang
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Obstacle field (arena size set at reset)
        self.obstacle_field = ObstacleField(
            arena_size=self._base_arena,
            robot_radius=self.robot_radius,
            obstacle_radius_range=tuple(self.cfg["obstacle_radius_range"]),
            dynamic_speed=float(self.cfg.get("dynamic_obstacle_speed", 0.15)),
            rng=self.rng,
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

    