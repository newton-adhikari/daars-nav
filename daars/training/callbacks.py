"""SB3 callbacks for logging navigation metrics during training."""

from __future__ import annotations

import sys
import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class NavCallback(BaseCallback):
    """
    Logs navigation-specific episode metrics with progress in console itself.
    """

    def __init__(
        self
    ):
        super().__init__()
