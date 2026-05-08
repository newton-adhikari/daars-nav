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

    def generate(self, num_static: int, num_dynamic: int = 0,
                 robot_pos: np.ndarray = None, goal_pos: np.ndarray = None) -> list[Obstacle]:
        
        # Generate non-overlapping obstacles avoiding robot start and goal
        self.obstacles = []
        exclusion_zones = []
        if robot_pos is not None:
            exclusion_zones.append((robot_pos, 1.0))
        if goal_pos is not None:
            exclusion_zones.append((goal_pos, 1.0))

        for i in range(num_static + num_dynamic):
            is_dyn = i >= num_static
            for _ in range(200):  # max placement attempts
                r = self.rng.uniform(self.r_min, self.r_max)
                margin = r + self.robot_radius + 0.3
                x = self.rng.uniform(margin, self.arena_size - margin)
                y = self.rng.uniform(margin, self.arena_size - margin)

                if self._is_valid_placement(x, y, r, exclusion_zones):
                    vx, vy = 0.0, 0.0
                    if is_dyn:
                        angle = self.rng.uniform(0, 2 * np.pi)
                        vx = self.dynamic_speed * np.cos(angle)
                        vy = self.dynamic_speed * np.sin(angle)
                    obs = Obstacle(x, y, r, is_dynamic=is_dyn, vx=vx, vy=vy)
                    self.obstacles.append(obs)
                    exclusion_zones.append((np.array([x, y]), r + 0.5))
                    break

        return self.obstacles

    def step(self, dt: float):
        # Update dynamic obstacle positions
        for obs in self.obstacles:
            if not obs.is_dynamic:
                continue
            obs.x += obs.vx * dt
            obs.y += obs.vy * dt
            margin = obs.radius + 0.1
            if obs.x < margin or obs.x > self.arena_size - margin:
                obs.vx *= -1
                obs.x = np.clip(obs.x, margin, self.arena_size - margin)
            if obs.y < margin or obs.y > self.arena_size - margin:
                obs.vy *= -1
                obs.y = np.clip(obs.y, margin, self.arena_size - margin)

    def _is_valid_placement(self, x, y, r, exclusion_zones) -> bool:
        pos = np.array([x, y])
        for ez_pos, ez_r in exclusion_zones:
            if np.linalg.norm(pos - ez_pos) < r + ez_r:
                return False
        for obs in self.obstacles:
            dist = np.sqrt((x - obs.x) ** 2 + (y - obs.y) ** 2)
            if dist < r + obs.radius + 0.3:
                return False
        return True

    def get_positions_and_radii(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.obstacles:
            return np.empty((0, 2)), np.empty(0)
        positions = np.array([[o.x, o.y] for o in self.obstacles])
        radii = np.array([o.radius for o in self.obstacles])
        return positions, radii
