"""Safety-Gymnasium wrapper for DAARS reward shaping.

Extracts goal distance and hazard proximity from the engine internals
(task.dist_goal(), task.agent.pos, task.hazards.pos) and applies
DAARS reward modulation via alpha/beta scaling functions.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class SafetyGymDAARSWrapper(gym.Wrapper):
    pass
