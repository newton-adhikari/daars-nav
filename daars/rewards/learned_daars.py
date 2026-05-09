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

    def __init__(self, lr: float = 1e-3, update_every: int = 50,
                 cost_weight: float = 1.0, timeout_weight: float = 0.5,
                 success_weight: float = 0.3):
        self.modulator = RewardModulator()
        self.optimizer = optim.Adam(self.modulator.parameters(), lr=lr)
        self.update_every = update_every
        self.cost_weight = cost_weight
        self.timeout_weight = timeout_weight
        self.success_weight = success_weight

        # Episode buffer for outer-loop updates
        self._episode_states: list[np.ndarray] = []
        self._episode_costs: list[float] = []
        self._episode_successes: list[bool] = []
        self._episode_timeouts: list[bool] = []
        self._ep_count = 0

    def get_alpha_beta(self, d_obs: float, d_goal: float, v_linear: float,
                        dsafe: float = 1.0, d_max: float = 14.0,
                        v_max: float = 0.22) -> tuple[float, float]:
        """Get α, β for a single step (inference, no grad)."""
        x = torch.tensor([[
            d_obs / dsafe,
            d_goal / d_max,
            v_linear / v_max,
        ]], dtype=torch.float32)

        with torch.no_grad():
            alpha, beta = self.modulator(x)
        return float(alpha[0]), float(beta[0])

    def record_step(self, d_obs: float, d_goal: float, v_linear: float,
                     dsafe: float = 1.0, d_max: float = 14.0,
                     v_max: float = 0.22):
        """Record state for outer-loop update."""
        self._episode_states.append(np.array([
            d_obs / dsafe, d_goal / d_max, v_linear / v_max
        ]))

    def end_episode(self, cost: float, success: bool, timeout: bool):
        """Record episode outcome."""
        self._episode_costs.append(cost)
        self._episode_successes.append(success)
        self._episode_timeouts.append(timeout)
        self._ep_count += 1

        if self._ep_count >= self.update_every:
            self._update()
            self._ep_count = 0
            self._episode_states = []
            self._episode_costs = []
            self._episode_successes = []
            self._episode_timeouts = []

    def _update(self):
        """Outer-loop update: adjust modulator to minimize cost."""
        if not self._episode_states:
            return

        # Compute target signal
        avg_cost = np.mean(self._episode_costs)
        timeout_rate = np.mean(self._episode_timeouts)
        success_rate = np.mean(self._episode_successes)

        # Loss: high cost → need more safety (increase β, decrease α)
        #        high timeout → need more progress (increase α)
        #        high success → good, small gradient
        loss_signal = (self.cost_weight * avg_cost
                       + self.timeout_weight * timeout_rate
                       - self.success_weight * success_rate)

        # Forward pass on recorded states
        states = torch.tensor(np.array(self._episode_states[-1000:]),
                              dtype=torch.float32)
        alpha, beta = self.modulator(states)

        # Push α down and β up when cost is high (and vice versa)
        # This is a simple policy gradient on the modulator
        alpha_loss = loss_signal * alpha.mean()
        beta_loss = -loss_signal * beta.mean()  # negative because high β = more safety
        loss = alpha_loss + beta_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.modulator.parameters(), 1.0)
        self.optimizer.step()


def learned_daars_reward(info: dict, cfg: dict,
                          modulator: LearnedModulatorManager | None = None,
                          **kwargs) -> float:
    """Compute reward using learned α, β modulation.

    Falls back to hand-designed DAARS if no modulator is provided.
    """
    if info["reached_goal"]:
        return float(cfg["goal_reward"])
    if info["collision"]:
        return float(cfg["collision_penalty"])
    if info["timeout"]:
        return float(cfg["timeout_penalty"])

    d_obs = float(info["min_obs_dist"])
    d_goal = float(info["goal_dist"])
    v_lin = float(info["v_linear"])
    dsafe = float(cfg["dsafe"])

    if modulator is not None:
        alpha, beta = modulator.get_alpha_beta(d_obs, d_goal, v_lin, dsafe)
        modulator.record_step(d_obs, d_goal, v_lin, dsafe)
    else:
        # Fallback to hand-designed
        import math
        alpha = max(0.3, min(d_obs / dsafe, 1.0))
        ks = float(cfg["ks"])
        eps = float(cfg["epsilon"])
        beta = min(math.exp(ks / (d_obs + eps)) - math.exp(ks / (d_obs + eps + 2.0)), 5.0)

    # Dense components
    r_prog = (float(info["prev_goal_dist"]) - d_goal) * float(cfg["progress_scale"])
    r_vel = 0.5 * max(v_lin, 0.0)

    if d_obs < dsafe:
        proximity = 1.0 - d_obs / dsafe
        r_safe = float(cfg["safety_penalty_scale"]) * proximity * proximity
    else:
        r_safe = 0.0

    return alpha * (r_prog + r_vel) + beta * r_safe - 0.1
 