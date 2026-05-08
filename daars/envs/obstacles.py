"""Obstacle generation for navigation environments."""

import numpy as np
from dataclasses import dataclass


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float
    is_dynamic: bool = False
    vx: float = 0.0
    vy: float = 0.0
