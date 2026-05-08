"""Obstacle generation for navigation environments."""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float
    is_dynamic: bool = False
    vx: float = 0.0
    vy: float = 0.0


class ObstacleField:
    """Generates and manages obstacle configurations."""

    def __init__(self, arena_size: float, robot_radius: float,
                 obstacle_radius_range: tuple = (0.2, 0.5),
                 dynamic_speed: float = 0.15, rng: np.random.Generator = None):
        self.arena_size = arena_size
        self.robot_radius = robot_radius
        self.r_min, self.r_max = obstacle_radius_range
        self.dynamic_speed = dynamic_speed
        self.rng = rng or np.random.default_rng()
        self.obstacles: list[Obstacle] = []

    
