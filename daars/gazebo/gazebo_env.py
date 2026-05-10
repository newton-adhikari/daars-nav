"""
Gazebo / ROS 2 environment bridge.
"""

from __future__ import annotations

import time
import threading
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist
    _ROS_AVAILABLE = True
except ImportError:
    _ROS_AVAILABLE = False

import gymnasium as gym
from gymnasium import spaces


class GazeboNavEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict,
        reward_fn=None,
        cost_fn=None,
        goal_pos: tuple[float, float] | None = None,
        step_duration: float | None = None,
    ):
        if not _ROS_AVAILABLE:
            raise RuntimeError(
                "ROS 2 is not available.  Source your ROS 2 workspace and "
                "install rclpy / sensor_msgs / nav_msgs / geometry_msgs."
            )

        super().__init__()
        self.cfg          = config["environment"]
        self.rcfg         = config["reward"]
        self.reward_fn    = reward_fn
        self.cost_fn      = cost_fn
        self._fixed_goal  = goal_pos
        self.step_dur     = step_duration or float(self.cfg["dt"])

        self.robot_radius  = float(self.cfg["robot_radius"])
        self.goal_threshold = float(self.cfg["goal_threshold"])
        self.num_rays      = int(self.cfg["num_lidar_rays"])
        self.lidar_range   = float(self.cfg["lidar_max_range"])
        self._max_lin      = float(self.cfg["max_linear_vel"])
        self._max_ang      = float(self.cfg["max_angular_vel"])
        self.arena         = float(self.cfg["arena_size"])
        self.max_steps     = int(self.cfg["max_steps"])

        self.action_space = spaces.Box(
            low=np.array([-self._max_lin, -self._max_ang], dtype=np.float32),
            high=np.array([ self._max_lin,  self._max_ang], dtype=np.float32),
        )
        obs_dim = self.num_rays + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ---- ROS 2 node ----
        rclpy.init(args=None)
        self._node = rclpy.create_node("daars_gazebo_env")

        self._pub_cmd = self._node.create_publisher(Twist, "/cmd_vel", 10)

        # Sensor data (updated by subscribers in a background thread)
        self._lock         = threading.Lock()
        self._scan_ranges  = np.full(self.num_rays, self.lidar_range)
        self._robot_x      = 0.0
        self._robot_y      = 0.0
        self._robot_yaw    = 0.0
        self._v_lin        = 0.0
        self._v_ang        = 0.0
        self._scan_ready   = False

        self._node.create_subscription(
            LaserScan, "/scan", self._scan_cb, 10)
        self._node.create_subscription(
            Odometry, "/odom", self._odom_cb, 10)

        self._ros_thread = threading.Thread(
            target=rclpy.spin, args=(self._node,), daemon=True)
        self._ros_thread.start()

        # Episode state
        self.goal_pos          = np.zeros(2)
        self.step_count        = 0
        self.prev_goal_dist    = 0.0
        self.start_goal_dist   = 0.0
        self.episode_min_clr   = float("inf")
        self.episode_path_len  = 0.0
        self.episode_cost_sum  = 0.0
        self._prev_pos         = np.zeros(2)

        # Normalisation
        self._inv_lidar      = 1.0 / self.lidar_range
        self._inv_arena_diag = 1.0 / (self.arena * np.sqrt(2))
        self._inv_pi         = 1.0 / np.pi

    
    # Subscriber callbacks (called from background ROS thread)

    def _scan_cb(self, msg: "LaserScan") -> None:
        ranges = np.array(msg.ranges, dtype=np.float32)
        # Resample to self.num_rays evenly spaced rays
        idx    = np.linspace(0, len(ranges) - 1, self.num_rays).astype(int)
        resampled = ranges[idx]
        np.clip(resampled, 0, self.lidar_range, out=resampled)
        # Replace inf/nan with max range
        resampled[~np.isfinite(resampled)] = self.lidar_range
        with self._lock:
            self._scan_ranges = resampled
            self._scan_ready  = True

    def _odom_cb(self, msg: "Odometry") -> None:
        pos = msg.pose.pose.position
        q   = msg.pose.pose.orientation
        # Yaw from quaternion
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw  = np.arctan2(siny, cosy)
        with self._lock:
            self._robot_x   = float(pos.x)
            self._robot_y   = float(pos.y)
            self._robot_yaw = float(yaw)
            self._v_lin     = float(msg.twist.twist.linear.x)
            self._v_ang     = float(msg.twist.twist.angular.z)
        

    
    # Gymnasium interface

    def reset(self, seed=None, options=None):
        # Stop robot
        self._publish_vel(0.0, 0.0)
        time.sleep(0.3)

        rng = np.random.default_rng(seed)

        # Teleport robot to a random safe spawn position
        sx, sy, syaw = self._find_safe_spawn(rng)
        self._teleport_robot(sx, sy, syaw)

        # Wait for fresh sensor data after respawn
        with self._lock:
            self._scan_ready = False
        for _ in range(150):          # up to 7.5 s
            with self._lock:
                ready = self._scan_ready
            if ready:
                break
            time.sleep(0.05)

        # Read (possibly updated) position from odom
        with self._lock:
            rx, ry = self._robot_x, self._robot_y

        # Set new random goal at least 2m away from spawn
        margin = 1.0
        self.goal_pos = np.array([4.0, 4.0])
        for _ in range(200):
            gp = rng.uniform(margin, self.arena - margin, size=2)
            if np.linalg.norm(gp - [rx, ry]) > 2.0:
                self.goal_pos = gp
                break

        self.start_goal_dist = float(np.linalg.norm(self.goal_pos - [rx, ry]))
        self.prev_goal_dist  = self.start_goal_dist
        self._prev_pos       = np.array([rx, ry])
        self.step_count      = 0
        self.episode_min_clr = float("inf")
        self.episode_path_len = 0.0
        self.episode_cost_sum = 0.0

        return self._obs(), self._info()

    # Known obstacle positions and radii from daars_arena.world
    OBSTACLES = [
        (1.5, 1.5, 0.40), (3.0, 1.3, 0.35), (4.5, 1.7, 0.45),
        (6.2, 1.4, 0.35), (2.0, 3.0, 0.40), (3.8, 2.8, 0.30),
        (5.5, 3.2, 0.45), (1.3, 4.5, 0.35), (3.2, 4.8, 0.40),
        (5.0, 4.3, 0.35), (6.5, 4.7, 0.30), (2.0, 6.0, 0.45),
        (4.0, 6.3, 0.35), (5.8, 5.8, 0.40), (6.8, 6.5, 0.30),
        (1.0, 2.5, 0.30),
    ]

    def _find_safe_spawn(self, rng: np.random.Generator) -> tuple:
        """Find a random spawn position that is clear of all obstacles.

        Returns (x, y, yaw) where the robot centre is at least
        (obstacle_radius + robot_radius + 0.15m) from every obstacle
        and at least 0.5m from every wall.
        """
        wall_margin = 0.5
        clearance   = self.robot_radius + 0.15  # extra buffer
        for _ in range(500):
            x = rng.uniform(wall_margin, self.arena - wall_margin)
            y = rng.uniform(wall_margin, self.arena - wall_margin)
            safe = True
            for ox, oy, r in self.OBSTACLES:
                if np.hypot(x - ox, y - oy) < r + clearance:
                    safe = False
                    break
            if safe:
                yaw = rng.uniform(-np.pi, np.pi)
                return x, y, yaw
        # Fallback: centre of arena
        return 4.0, 4.0, 0.0

    def _teleport_robot(self, x: float, y: float, yaw: float):
        """Delete and respawn the robot at (x, y, yaw) via subprocess.

        Uses the Gazebo /delete_entity + spawn_entity.py workflow which
        is the most reliable method under WSL2 / headless Gazebo.
        """
        self._publish_vel(0.0, 0.0)
        time.sleep(0.2)

        import subprocess
        try:
            subprocess.run(
                ["ros2", "service", "call", "/delete_entity",
                 "gazebo_msgs/srv/DeleteEntity", "{name: 'burger'}"],
                capture_output=True, timeout=5,
            )
            time.sleep(0.5)

            sdf_path = "/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf"
            result = subprocess.run(
                ["ros2", "run", "gazebo_ros", "spawn_entity.py",
                 "-entity", "burger", "-file", sdf_path,
                 "-x", str(x), "-y", str(y), "-z", "0.01",
                 "-Y", str(yaw)],
                capture_output=True, timeout=10,
            )
            if result.returncode == 0:
                self._node.get_logger().info(
                    f"Respawned at ({x:.1f}, {y:.1f}, yaw={yaw:.2f})")
                time.sleep(1.0)  # wait for plugins to reconnect
                with self._lock:
                    self._scan_ready = False
            else:
                self._node.get_logger().warn(
                    f"Spawn failed: {result.stderr.decode()}")
        except Exception as e:
            self._node.get_logger().warn(f"Teleport failed: {e}")

    def _move_goal_marker(self, x: float, y: float):
        # Move the visual goal marker. Silent on failure
        import subprocess
        try:
            subprocess.run(["ros2", "service", "call", "/delete_entity",
                           "gazebo_msgs/srv/DeleteEntity", "{name: 'goal_marker'}"],
                          capture_output=True, timeout=2)
        except Exception:
            pass
        try:
            sdf = '<?xml version="1.0"?><sdf version="1.6"><model name="goal_marker"><static>true</static><link name="l"><visual name="v"><geometry><cylinder><radius>0.15</radius><length>0.02</length></cylinder></geometry><material><ambient>0.2 0.9 0.2 1</ambient></material></visual></link></model></sdf>'
            with open("/tmp/goal_marker.sdf", "w") as f:
                f.write(sdf)
            subprocess.run(["ros2", "run", "gazebo_ros", "spawn_entity.py",
                           "-entity", "goal_marker", "-file", "/tmp/goal_marker.sdf",
                           "-x", str(x), "-y", str(y), "-z", "0.01"],
                          capture_output=True, timeout=3)
        except Exception:
            pass

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._publish_vel(float(action[0]), float(action[1]))
        time.sleep(self.step_dur)

        with self._lock:
            rx, ry    = self._robot_x, self._robot_y
            yaw       = self._robot_yaw
            v_lin     = self._v_lin
            v_ang     = self._v_ang
            scan      = self._scan_ranges.copy()

        pos         = np.array([rx, ry])
        goal_dist   = float(np.linalg.norm(self.goal_pos - pos))
        min_obs_dist = float(np.min(scan)) - self.robot_radius

        if min_obs_dist < self.episode_min_clr:
            self.episode_min_clr = min_obs_dist
        self.episode_path_len += float(np.linalg.norm(pos - self._prev_pos))
        self._prev_pos = pos.copy()
        self.step_count += 1

        reached_goal = goal_dist < self.goal_threshold
        collision    = min_obs_dist < 0.0  # LiDAR reading closer than robot radius
        timeout      = self.step_count >= self.max_steps
        terminated   = reached_goal or collision
        truncated    = timeout and not terminated

        step_info = {
            "goal_dist":      goal_dist,
            "prev_goal_dist": self.prev_goal_dist,
            "min_obs_dist":   max(min_obs_dist, 0.0),
            "reached_goal":   reached_goal,
            "collision":      collision,
            "timeout":        timeout,
            "v_linear":       v_lin,
        }
        reward = self.reward_fn(step_info, self.rcfg) if self.reward_fn else 0.0
        cost   = self.cost_fn(step_info, self.rcfg)   if self.cost_fn  else 0.0
        self.episode_cost_sum += cost
        self.prev_goal_dist    = goal_dist

        if terminated or truncated:
            self._publish_vel(0.0, 0.0)

        info = self._info(reached_goal=reached_goal, collision=collision)
        info["cost"] = cost
        return self._obs(), reward, terminated, truncated, info

    
    # Helpers
    
    def _publish_vel(self, v_lin: float, v_ang: float) -> None:
        msg = Twist()
        msg.linear.x  = float(np.clip(v_lin, -self._max_lin, self._max_lin))
        msg.angular.z = float(np.clip(v_ang, -self._max_ang, self._max_ang))
        self._pub_cmd.publish(msg)

    def _obs(self) -> np.ndarray:
        with self._lock:
            ranges = self._scan_ranges.copy()
            rx, ry = self._robot_x, self._robot_y
            yaw    = self._robot_yaw
            v_lin  = self._v_lin
            v_ang  = self._v_ang

        obs = np.empty(self.num_rays + 4, dtype=np.float32)
        obs[:self.num_rays] = ranges * self._inv_lidar
        gx = self.goal_pos[0] - rx
        gy = self.goal_pos[1] - ry
        obs[self.num_rays]     = np.sqrt(gx*gx + gy*gy) * self._inv_arena_diag
        ga                     = np.arctan2(gy, gx) - yaw
        obs[self.num_rays + 1] = ((ga + np.pi) % (2*np.pi) - np.pi) * self._inv_pi
        obs[self.num_rays + 2] = v_lin / self._max_lin
        obs[self.num_rays + 3] = v_ang / self._max_ang
        return obs

    def _info(self, reached_goal=False, collision=False) -> dict:
        with self._lock:
            rx, ry = self._robot_x, self._robot_y
        return {
            "goal_dist":       float(np.linalg.norm(self.goal_pos - [rx, ry])),
            "min_clearance":   self.episode_min_clr,
            "path_length":     self.episode_path_len,
            "start_goal_dist": self.start_goal_dist,
            "cumulative_cost": self.episode_cost_sum,
            "reached_goal":    reached_goal,
            "collision":       collision,
            "steps":           self.step_count,
        }
    
    def close(self):
        self._publish_vel(0.0, 0.0)
        self._node.destroy_node()
        rclpy.shutdown()
 