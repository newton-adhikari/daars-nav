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

import torch.nn as nn


class RewardModulator(nn.Module):
    
    def __init__(self, input_dim: int = 3, hidden: int = 16):
        super().__init__()
        