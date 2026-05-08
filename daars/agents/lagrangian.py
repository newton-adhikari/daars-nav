"""
Lagrangian RL agents for safe navigation.

PPO-Lagrangian and SAC-Lagrangian are the standard baselines in safe
reinforcement learning (Garcia & Fernandez 2015; Ray et al. 2019
Safety Gym paper). 

    max_π  J_R(π) − λ · J_C(π)
    s.t.   J_C(π) ≤ d   (cost limit)

The multiplier λ is updated by dual ascent:
    λ ← max(0, λ + α_λ · (J_C − d))
"""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


# try to create shared Lagrangian multiplier state
# TODO: may need to update later


class LagrangianMultiplier:

    def __init__(
        self,
    ):
        self.value      = 0
        self.lr         = 0
        self.cost_limit = 0
        self.max_value  = 0
        self._costs: list[float] = []
        self._window    = 0


# defined a callback to, injects Lagrangian penalty into rewards and updates λ

class LagrangianCallback(BaseCallback):
    """
    SB3 callback that implements the Lagrangian reward adjustment.
    """

    def __init__(
        self,
    ):
        super().__init__(0)
        


# PPO-Lagrangian

def make_ppo_lagrangian(
    
) -> tuple[PPO, LagrangianCallback]:
    """
    Construct a PPO model and with Lagrangian callback.
    """
    pass


# SAC-Lagrangian

def make_sac_lagrangian(
    
) -> tuple[SAC, LagrangianCallback]:
    """
    Construct a SAC model for the Lagrangian constrained baseline.
    """
 