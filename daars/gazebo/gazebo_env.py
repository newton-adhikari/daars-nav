"""
Gazebo / ROS 2 environment bridge.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import rclpy


class GazeboNavEnv(gym.Env):

    def __init__(
        self,
    ):
       
        super().__init__()
        

    def close(self):
        self._publish_vel(0.0, 0.0)
        self._node.destroy_node()
        rclpy.shutdown()
 