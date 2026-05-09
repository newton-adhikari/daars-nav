"""
Gazebo / ROS 2 environment bridge.
"""

from __future__ import annotations

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
        

    def close(self):
        self._publish_vel(0.0, 0.0)
        self._node.destroy_node()
        rclpy.shutdown()
 