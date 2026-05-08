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

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

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
        self.start_goal_dist    = 0.0   # straight-line dist at reset
        self.episode_min_clearance = float("inf")
        self.episode_path_length   = 0.0
        self.prev_pos           = np.zeros(2)
        self.episode_cost_sum   = 0.0

        # Obstacle cache
        self._obs_positions  = np.empty((0, 2))
        self._obs_radii      = np.empty(0)
        self._obs_radii_sq   = np.empty(0)
        self._has_obstacles  = False


    # Gymnasium interface

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Domain randomise arena size
        if self.domain_rand:
            self.arena = float(self.rng.uniform(*self._arena_range))
            self.obstacle_field.arena_size = self.arena
        else:
            self.arena = self._base_arena

        self._inv_arena_diag = 1.0 / (self.arena * np.sqrt(2))

        margin = 1.5
        # Robot start
        self.robot_pos   = self.rng.uniform(margin, self.arena - margin, size=2)
        self.robot_theta = self.rng.uniform(-np.pi, np.pi)
        self.robot_vel[:] = 0.0

        # Goal — at least 3 m from robot
        for _ in range(200):
            self.goal_pos = self.rng.uniform(margin, self.arena - margin, size=2)
            if np.linalg.norm(self.goal_pos - self.robot_pos) > 3.0:
                break

        # Store straight-line distance at episode start (used for path efficiency)
        self.start_goal_dist = float(np.linalg.norm(self.goal_pos - self.robot_pos))

        # Obstacles
        n_static, n_dynamic = self._obstacle_counts()
        self.obstacle_field.rng = self.rng
        self.obstacle_field.generate(
            n_static, n_dynamic,
            robot_pos=self.robot_pos,
            goal_pos=self.goal_pos,
        )
        self._cache_obstacles()

        self.step_count              = 0
        self.prev_goal_dist          = self.start_goal_dist
        self.episode_min_clearance   = float("inf")
        self.episode_path_length     = 0.0
        self.episode_cost_sum        = 0.0
        self.prev_pos[:] = self.robot_pos

        # Dynamics domain randomisation: randomise per episode
        if self.domain_rand:
            lo, hi = self._action_delay_range
            self._action_delay_steps = int(self.rng.integers(lo, hi + 1))
            lo, hi = self._vel_smoothing_range
            self._vel_smooth_alpha = float(self.rng.uniform(lo, hi))
        else:
            self._action_delay_steps = 0
            self._vel_smooth_alpha = 1.0
        self._action_buffer = []
        self._smooth_v_lin = 0.0
        self._smooth_v_ang = 0.0

        return self._obs(), self._info()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Add action noise (domain randomisation)
        if self.domain_rand and self._vel_noise_std > 0:
            action = action + self.rng.normal(0, self._vel_noise_std, size=2)
            action = np.clip(action, self.action_space.low, self.action_space.high)

        # Action delay: buffer actions, execute delayed (simulates ROS latency)
        if self.domain_rand and self._action_delay_steps > 0:
            self._action_buffer.append(action.copy())
            if len(self._action_buffer) > self._action_delay_steps:
                action = self._action_buffer.pop(0)
            else:
                action = np.zeros(2)

        v_lin_cmd = float(action[0])
        v_ang_cmd = float(action[1])

        # Velocity smoothing: EMA simulating inertia
        alpha = self._vel_smooth_alpha
        self._smooth_v_lin = alpha * v_lin_cmd + (1 - alpha) * self._smooth_v_lin
        self._smooth_v_ang = alpha * v_ang_cmd + (1 - alpha) * self._smooth_v_ang
        v_lin = self._smooth_v_lin
        v_ang = self._smooth_v_ang

        # Differential-drive kinematics
        self.robot_theta += v_ang * self.dt
        self.robot_theta  = (self.robot_theta + np.pi) % (2 * np.pi) - np.pi
        dx = v_lin * np.cos(self.robot_theta) * self.dt
        dy = v_lin * np.sin(self.robot_theta) * self.dt
        self.robot_pos[0] += dx
        self.robot_pos[1] += dy
        self.robot_vel[0]  = v_lin
        self.robot_vel[1]  = v_ang

        # Wall collision (clamp + flag)
        prev_pos_snap = self.robot_pos.copy()
        np.clip(self.robot_pos, self.robot_radius,
                self.arena - self.robot_radius, out=self.robot_pos)
        wall_hit = not np.allclose(self.robot_pos, prev_pos_snap)

        # Dynamic obstacles
        if any(o.is_dynamic for o in self.obstacle_field.obstacles):
            self.obstacle_field.step(self.dt)
            self._cache_obstacles()

        # Distances
        goal_dist   = float(np.linalg.norm(self.goal_pos - self.robot_pos))
        min_obs_dist = self._min_obs_dist()

        # Tracking
        min_obs_dist = min(min_obs_dist, 0.0 if wall_hit else float("inf"))
        if min_obs_dist < self.episode_min_clearance:
            self.episode_min_clearance = min_obs_dist
        self.episode_path_length += float(np.linalg.norm(self.robot_pos - self.prev_pos))
        self.prev_pos[:] = self.robot_pos
        self.step_count  += 1

        # Terminal conditions
        reached_goal = goal_dist < self.goal_threshold
        collision    = (min_obs_dist < self.robot_radius) or wall_hit
        timeout      = self.step_count >= self.max_steps
        terminated   = reached_goal or collision
        truncated    = timeout and not terminated

        # Reward and cost
        step_info = {
            "goal_dist":      goal_dist,
            "prev_goal_dist": self.prev_goal_dist,
            "min_obs_dist":   max(min_obs_dist, 0.0),
            "reached_goal":   reached_goal,
            "collision":      collision,
            "timeout":        timeout,
            "v_linear":       v_lin,
            "lidar_ranges":   self._ranges_buf.copy(),
            "goal_angle":     float(((np.arctan2(
                self.goal_pos[1] - self.robot_pos[1],
                self.goal_pos[0] - self.robot_pos[0])
                - self.robot_theta + np.pi) % (2 * np.pi) - np.pi)),
        }
        reward = self.reward_fn(step_info, self.rcfg) if self.reward_fn else 0.0
        cost   = self.cost_fn(step_info, self.rcfg)   if self.cost_fn  else 0.0
        self.episode_cost_sum += cost
        self.prev_goal_dist    = goal_dist

        info = self._info(reached_goal=reached_goal, collision=collision)
        info["cost"] = cost   # for lagrangian agents they need to read cost 

        return self._obs(), reward, terminated, truncated, info

    # create helpers

    def _obstacle_counts(self) -> tuple[int, int]:
        c = self.cfg
        if self.scenario == "simple":
            lo, hi = c["simple_obstacles"]
            return int(self.rng.integers(lo, hi + 1)), 0
        if self.scenario == "complex":
            lo, hi = c["complex_obstacles"]
            return int(self.rng.integers(lo, hi + 1)), 0
        if self.scenario == "dynamic":
            sl, sh = c["simple_obstacles"]
            dl, dh = c["dynamic_obstacles"]
            return (int(self.rng.integers(sl, sh + 1)),
                    int(self.rng.integers(dl, dh + 1)))
        return 3, 0
    
    def _cache_obstacles(self):
        self._obs_positions, self._obs_radii = \
            self.obstacle_field.get_positions_and_radii()
        self._has_obstacles = len(self._obs_positions) > 0
        if self._has_obstacles:
            self._obs_radii_sq = self._obs_radii ** 2

    def _min_obs_dist(self) -> float:
        rx, ry = self.robot_pos
        wall_min = min(rx, ry, self.arena - rx, self.arena - ry)
        if not self._has_obstacles:
            return wall_min
        dists = np.linalg.norm(self._obs_positions - self.robot_pos, axis=1)
        dists -= self._obs_radii
        return min(float(np.min(dists)), wall_min)
    
    def _obs(self) -> np.ndarray:
        ranges = self._lidar()
        buf    = self._obs_buf
        nr     = self.num_rays
        buf[:nr] = ranges * self._inv_lidar

        gx = self.goal_pos[0] - self.robot_pos[0]
        gy = self.goal_pos[1] - self.robot_pos[1]
        buf[nr]     = np.sqrt(gx*gx + gy*gy) * self._inv_arena_diag
        ga          = np.arctan2(gy, gx) - self.robot_theta
        buf[nr + 1] = ((ga + np.pi) % (2 * np.pi) - np.pi) * self._inv_pi
        buf[nr + 2] = self.robot_vel[0] / self._max_lin
        buf[nr + 3] = self.robot_vel[1] / self._max_ang
        return buf.copy()
    
    def _info(self, reached_goal=False, collision=False) -> dict:
        return {
            "goal_dist":         float(np.linalg.norm(self.goal_pos - self.robot_pos)),
            "min_clearance":     self.episode_min_clearance,
            "path_length":       self.episode_path_length,
            "start_goal_dist":   self.start_goal_dist,   # ← correct for efficiency
            "cumulative_cost":   self.episode_cost_sum,
            "reached_goal":      reached_goal,
            "collision":         collision,
            "steps":             self.step_count,
        }