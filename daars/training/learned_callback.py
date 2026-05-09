"""
Callback for training with learned reward modulation.
"""

from __future__ import annotations
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class LearnedDAARSCallback(BaseCallback):
    """SB3 callback that manages the learned reward modulator."""

    def __init__(
        self,
    ):
        super().__init__()
