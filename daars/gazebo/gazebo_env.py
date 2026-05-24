"""
Gazebo / ROS 2 environment bridge for TurtleBot3 Burger.

Design goals:
  - Fast resets via /gazebo/set_model_state topic (no delete/respawn)
  - Proper thread safety for ROS callbacks
  - Compatible with SB3 model.predict() interface
  - Publishable-quality metrics collection
"""

from __future__ import annotations

import time
import threading
import subprocess
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist
    from gazebo_msgs.msg import ModelState
    _ROS_AVAILABLE = True
except ImportError:
    _ROS_AVAILABLE = False

import gymnasium as gym
from gymnasium import spaces


# Obstacle positions and radii from daars_arena.world (16 cylinders)
ARENA_OBSTACLES = [
    (1.5, 1.5, 0.40), (3.0, 1.3, 0.35), (4.5, 1.7, 0.45),
    (6.2, 1.4, 0.35), (2.0, 3.0, 0.40), (3.8, 2.8, 0.30),
    (5.5, 3.2, 0.45), (1.3, 4.5, 0.35), (3.2, 4.8, 0.40),
    (5.0, 4.3, 0.35), (6.5, 4.7, 0.30), (2.0, 6.0, 0.45),
    (4.0, 6.3, 0.35), (5.8, 5.8, 0.40), (6.8, 6.5, 0.30),
    (1.0, 2.5, 0.30),
]


class GazeboNavEnv(gym.Env):
    """Gymnasium environment wrapping Gazebo + TurtleBot3 via ROS 2.

    Key design choice: resets use /gazebo/set_model_state topic to teleport
    the robot WITHOUT deleting/respawning it. This preserves all ROS topic
    connections (/cmd_vel, /odom, /scan) across episodes.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict,
        reward_fn=None,
        cost_fn=None,
        goal_pos: tuple[float, float] | None = None,
        fast_mode: bool = True,
    ):
        if not _ROS_AVAILABLE:
            raise RuntimeError(
                "ROS 2 not available. Source your ROS 2 workspace first.\n"
                "  source /opt/ros/humble/setup.bash"
            )

        super().__init__()
        self.cfg = config["environment"]
        self.rcfg = config["reward"]
        self.reward_fn = reward_fn
        self.cost_fn = cost_fn
        self._fixed_goal = np.array(goal_pos) if goal_pos else None

        # Environment parameters
        self.robot_radius = float(self.cfg["robot_radius"])
        self.goal_threshold = float(self.cfg["goal_threshold"])
        self.num_rays = int(self.cfg["num_lidar_rays"])
        self.lidar_range = float(self.cfg["lidar_max_range"])
        self._max_lin = float(self.cfg["max_linear_vel"])
        self._max_ang = float(self.cfg["max_angular_vel"])
        self.arena = float(self.cfg["arena_size"])
        self.max_steps = int(self.cfg["max_steps"])

        # Step timing
        self.step_dur = 0.05 if fast_mode else float(self.cfg["dt"])

        # Spaces
        self.action_space = spaces.Box(
            low=np.array([-self._max_lin, -self._max_ang], dtype=np.float32),
            high=np.array([self._max_lin, self._max_ang], dtype=np.float32),
        )
        obs_dim = self.num_rays + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ROS 2 setup
        if not rclpy.ok():
            rclpy.init(args=None)

        self._node = rclpy.create_node("daars_gazebo_env")
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)

        # Publishers
        self._pub_cmd = self._node.create_publisher(Twist, "/cmd_vel", 10)
        self._pub_model_state = self._node.create_publisher(
            ModelState, "/gazebo/set_model_state", 10
        )

        # Sensor state (updated by callbacks in background thread)
        self._lock = threading.Lock()
        self._scan_ranges = np.full(self.num_rays, self.lidar_range, dtype=np.float32)
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_yaw = 0.0
        self._v_lin = 0.0
        self._v_ang = 0.0
        self._scan_seq = 0

        self._node.create_subscription(LaserScan, "/scan", self._scan_cb, 10)
        self._node.create_subscription(Odometry, "/odom", self._odom_cb, 10)

        # Spin ROS in background
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True
        )
        self._spin_thread.start()

        # Episode state
        self.goal_pos = np.zeros(2)
        self.step_count = 0
        self.prev_goal_dist = 0.0
        self.start_goal_dist = 0.0
        self.episode_min_clr = float("inf")
        self.episode_path_len = 0.0
        self.episode_cost_sum = 0.0
        self._prev_pos = np.zeros(2)

        # Precomputed normalisation constants
        self._inv_lidar = 1.0 / self.lidar_range
        self._inv_arena_diag = 1.0 / (self.arena * np.sqrt(2))
        self._inv_pi = 1.0 / np.pi

        # Wait for initial sensor data
        self._wait_for_scan(timeout=10.0)

    # ─── ROS Callbacks ───────────────────────────────────────────────────

    def _scan_cb(self, msg: "LaserScan") -> None:
        ranges = np.array(msg.ranges, dtype=np.float32)
        idx = np.linspace(0, len(ranges) - 1, self.num_rays).astype(int)
        resampled = ranges[idx]
        np.clip(resampled, 0, self.lidar_range, out=resampled)
        resampled[~np.isfinite(resampled)] = self.lidar_range
        with self._lock:
            self._scan_ranges = resampled
            self._scan_seq += 1

    def _odom_cb(self, msg: "Odometry") -> None:
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = np.arctan2(siny, cosy)
        with self._lock:
            self._robot_x = float(pos.x)
            self._robot_y = float(pos.y)
            self._robot_yaw = float(yaw)
            self._v_lin = float(msg.twist.twist.linear.x)
            self._v_ang = float(msg.twist.twist.angular.z)

    # ─── Gymnasium Interface ─────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        # Stop robot movement
        self._stop_robot()
        time.sleep(0.1)

        rng = np.random.default_rng(seed)

        # Teleport to safe spawn (no delete/respawn — preserves topic connections)
        sx, sy, syaw = self._find_safe_spawn(rng)
        self._teleport(sx, sy, syaw)

        # CRITICAL: wait for the scan to update AFTER teleport.
        # The scan from the old position is stale and will cause false collisions.
        # We need at least 2 fresh scans to be sure the data reflects new position.
        self._wait_for_fresh_scans(n=2, timeout=3.0)

        # Verify we're not in collision at spawn (scan-based check)
        with self._lock:
            scan = self._scan_ranges.copy()
            rx, ry = self._robot_x, self._robot_y

        min_scan = float(np.min(scan))
        if min_scan < self.robot_radius + 0.05:
            # We spawned too close to something — try again with a different position
            for retry in range(5):
                sx2, sy2, syaw2 = self._find_safe_spawn(rng)
                self._teleport(sx2, sy2, syaw2)
                self._wait_for_fresh_scans(n=2, timeout=2.0)
                with self._lock:
                    scan = self._scan_ranges.copy()
                    rx, ry = self._robot_x, self._robot_y
                if float(np.min(scan)) >= self.robot_radius + 0.05:
                    break

        # Set goal
        if self._fixed_goal is not None:
            self.goal_pos = self._fixed_goal.copy()
        else:
            self.goal_pos = self._random_goal(rng, rx, ry)

        self.start_goal_dist = float(np.linalg.norm(self.goal_pos - [rx, ry]))
        self.prev_goal_dist = self.start_goal_dist
        self._prev_pos = np.array([rx, ry])
        self.step_count = 0
        self.episode_min_clr = float("inf")
        self.episode_path_len = 0.0
        self.episode_cost_sum = 0.0

        return self._obs(), self._info()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._publish_vel(float(action[0]), float(action[1]))
        time.sleep(self.step_dur)

        with self._lock:
            rx, ry = self._robot_x, self._robot_y
            yaw = self._robot_yaw
            v_lin = self._v_lin
            scan = self._scan_ranges.copy()

        pos = np.array([rx, ry])
        goal_dist = float(np.linalg.norm(self.goal_pos - pos))
        min_obs_dist = float(np.min(scan)) - self.robot_radius

        # Track metrics
        self.episode_min_clr = min(self.episode_min_clr, min_obs_dist)
        self.episode_path_len += float(np.linalg.norm(pos - self._prev_pos))
        self._prev_pos = pos.copy()
        self.step_count += 1

        # Termination conditions
        reached_goal = goal_dist < self.goal_threshold
        collision = min_obs_dist < 0.0
        timeout = self.step_count >= self.max_steps
        terminated = reached_goal or collision
        truncated = timeout and not terminated

        # Reward
        step_info = {
            "goal_dist": goal_dist,
            "prev_goal_dist": self.prev_goal_dist,
            "min_obs_dist": max(min_obs_dist, 0.0),
            "reached_goal": reached_goal,
            "collision": collision,
            "timeout": timeout,
            "v_linear": v_lin,
        }
        reward = self.reward_fn(step_info, self.rcfg) if self.reward_fn else 0.0
        cost = self.cost_fn(step_info, self.rcfg) if self.cost_fn else 0.0
        self.episode_cost_sum += cost
        self.prev_goal_dist = goal_dist

        if terminated or truncated:
            self._stop_robot()

        info = self._info(reached_goal=reached_goal, collision=collision)
        info["cost"] = cost
        return self._obs(), reward, terminated, truncated, info

    def close(self):
        self._stop_robot()
        try:
            self._executor.shutdown()
            self._node.destroy_node()
        except Exception:
            pass

    # ─── Teleportation (NO delete/respawn) ───────────────────────────────

    def _teleport(self, x: float, y: float, yaw: float):
        """Teleport robot using `gz model` CLI command.

        This moves the model in-place without deleting/respawning, so all
        ROS topic connections remain intact. The `gz model` command talks
        directly to Gazebo's transport layer and is reliable.
        """
        # Stop any movement first
        self._stop_robot()
        time.sleep(0.05)

        # Use gz model command to set pose directly
        # This bypasses ROS entirely and talks to Gazebo's internal transport
        qz = float(np.sin(yaw / 2.0))
        qw = float(np.cos(yaw / 2.0))

        try:
            subprocess.run(
                [
                    "gz", "model", "-m", "burger",
                    "-x", str(x), "-y", str(y), "-z", "0.01",
                    "-R", "0", "-P", "0", "-Y", str(yaw),
                ],
                capture_output=True, timeout=3.0,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Fallback: try ros2 service call
            self._teleport_service(x, y, yaw)
            return

        # Also zero the velocity via the model state topic (belt and suspenders)
        msg = ModelState()
        msg.model_name = "burger"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.01
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        msg.twist = Twist()
        msg.reference_frame = "world"
        self._pub_model_state.publish(msg)

        time.sleep(0.3)

    def _teleport_service(self, x: float, y: float, yaw: float):
        """Fallback: use ros2 service call to set entity state."""
        qz = float(np.sin(yaw / 2.0))
        qw = float(np.cos(yaw / 2.0))
        yaml_str = (
            f"{{state: {{name: 'burger', "
            f"pose: {{position: {{x: {x}, y: {y}, z: 0.01}}, "
            f"orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}}}, "
            f"twist: {{linear: {{x: 0, y: 0, z: 0}}, angular: {{x: 0, y: 0, z: 0}}}}}}}}"
        )
        try:
            subprocess.run(
                ["ros2", "service", "call", "/set_entity_state",
                 "gazebo_msgs/srv/SetEntityState", yaml_str],
                capture_output=True, timeout=5.0,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        time.sleep(0.3)

    def _wait_for_fresh_scans(self, n: int = 2, timeout: float = 3.0):
        """Wait for N fresh scan messages after teleport.

        This ensures the LiDAR data reflects the robot's NEW position,
        not stale data from before the teleport.
        """
        with self._lock:
            start_seq = self._scan_seq
        target_seq = start_seq + n
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._scan_seq >= target_seq:
                    return
            time.sleep(0.02)

    # ─── Helpers ─────────────────────────────────────────────────────────

    def _find_safe_spawn(self, rng: np.random.Generator) -> tuple:
        """Random spawn position clear of all obstacles and walls.

        Uses a generous clearance buffer (robot_radius + 0.4m) to account for
        Gazebo collision geometry being slightly larger than visual geometry.
        """
        wall_margin = 0.8
        clearance = self.robot_radius + 0.40  # generous buffer
        for _ in range(500):
            x = rng.uniform(wall_margin, self.arena - wall_margin)
            y = rng.uniform(wall_margin, self.arena - wall_margin)
            safe = True
            for ox, oy, r in ARENA_OBSTACLES:
                if np.hypot(x - ox, y - oy) < r + clearance:
                    safe = False
                    break
            if safe:
                return x, y, rng.uniform(-np.pi, np.pi)
        return 4.0, 4.0, 0.0  # fallback: center

    def _random_goal(self, rng: np.random.Generator, rx: float, ry: float) -> np.ndarray:
        """Random goal at least 2m from robot, clear of obstacles."""
        margin = 0.8
        clearance = 0.5
        for _ in range(300):
            gx = rng.uniform(margin, self.arena - margin)
            gy = rng.uniform(margin, self.arena - margin)
            if np.hypot(gx - rx, gy - ry) < 2.0:
                continue
            safe = True
            for ox, oy, r in ARENA_OBSTACLES:
                if np.hypot(gx - ox, gy - oy) < r + clearance:
                    safe = False
                    break
            if safe:
                return np.array([gx, gy])
        return np.array([self.arena - 1.0, self.arena - 1.0])

    def _wait_for_scan(self, timeout: float = 5.0):
        """Block until a new scan arrives."""
        with self._lock:
            start_seq = self._scan_seq
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._scan_seq > start_seq:
                    return
            time.sleep(0.02)

    def _stop_robot(self):
        """Publish zero velocity."""
        self._publish_vel(0.0, 0.0)

    def _publish_vel(self, v_lin: float, v_ang: float):
        msg = Twist()
        msg.linear.x = float(np.clip(v_lin, -self._max_lin, self._max_lin))
        msg.angular.z = float(np.clip(v_ang, -self._max_ang, self._max_ang))
        self._pub_cmd.publish(msg)

    def _obs(self) -> np.ndarray:
        with self._lock:
            ranges = self._scan_ranges.copy()
            rx, ry = self._robot_x, self._robot_y
            yaw = self._robot_yaw
            v_lin = self._v_lin
            v_ang = self._v_ang

        obs = np.empty(self.num_rays + 4, dtype=np.float32)
        obs[:self.num_rays] = ranges * self._inv_lidar

        gx = self.goal_pos[0] - rx
        gy = self.goal_pos[1] - ry
        obs[self.num_rays] = np.sqrt(gx * gx + gy * gy) * self._inv_arena_diag
        ga = np.arctan2(gy, gx) - yaw
        obs[self.num_rays + 1] = ((ga + np.pi) % (2 * np.pi) - np.pi) * self._inv_pi
        obs[self.num_rays + 2] = v_lin / self._max_lin
        obs[self.num_rays + 3] = v_ang / self._max_ang
        return obs

    def _info(self, reached_goal=False, collision=False) -> dict:
        with self._lock:
            rx, ry = self._robot_x, self._robot_y
        return {
            "goal_dist": float(np.linalg.norm(self.goal_pos - [rx, ry])),
            "min_clearance": self.episode_min_clr,
            "path_length": self.episode_path_len,
            "start_goal_dist": self.start_goal_dist,
            "cumulative_cost": self.episode_cost_sum,
            "reached_goal": reached_goal,
            "collision": collision,
            "steps": self.step_count,
        }
 