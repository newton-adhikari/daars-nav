"""
Learned DAARS: learned reward modulation.

 α(d) and β(d) are heuristic — they assume a specific
functional relationship between obstacle distance and optimal reward
weights. Learned DAARS replaces these with a small neural network
that outputs [α, β] conditioned on the local safety state.

we will train modulator network with a bilevel objective:
  - Inner loop: PPO trains the policy using modulated rewards
  - Outer loop: The modulator is updated to minimize episode cost
    (collisions + proximity violations)

The modulator learns WHEN to suppress progress and WHEN to amplify
safety, adapting to the specific environment geometry rather than
relying on a universal functional form.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class RewardModulator(nn.Module):
    """Small MLP that outputs [α, β] given safety state.

    Input: [d_obs/d_safe, d_goal/d_max, v_linear/v_max]
    Output: [α ∈ [α_min, 1], β ∈ [0, β_max]]

    Architecture is made small (< 500 params) to avoid
    overfitting and to have interpretable post-hoc.
    """

    ALPHA_MIN = 0.3
    BETA_MAX = 5.0

    def __init__(self, input_dim: int = 3, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )
        # Initialize to approximate hand-designed DAARS
        self._init_weights()
        
    def _init_weights(self):
        # Initialize so initial behavior ≈ hand-designed DAARS.
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        # α: sigmoid scaled to [α_min, 1]
        alpha = self.ALPHA_MIN + (1.0 - self.ALPHA_MIN) * torch.sigmoid(out[:, 0])
        # β: sigmoid scaled to [0, β_max]
        beta = self.BETA_MAX * torch.sigmoid(out[:, 1])
        return alpha, beta
    

class LearnedModulatorManager:
    """Manages the modulator network training alongside PPO."""

    def __init__(self):pass
 