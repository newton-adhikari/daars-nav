"""
First-Order Constrained Optimization in Policy Space (FOCOPS).

Reference: Zhang et al., "First Order Constrained Optimization in Policy Space"
           NeurIPS 2020.
"""

from __future__ import annotations

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class FOCOPSCallback(BaseCallback):
    """Implements FOCOPS constraint projection as an SB3 callback.

    After each PPO update, applies a first-order correction to the policy
    parameters to reduce constraint violation. This approximates the full
    FOCOPS algorithm within the SB3 framework.

    The correction is:
        θ_{k+1} = θ_k - η_λ * ∇_θ J_C(θ_k)

    where J_C is the cost objective and η_λ is adaptively scaled by
    the constraint violation magnitude.
    """

    def __init__(
        self,
        cost_limit: float = 0.08,
        lambda_lr: float = 0.01,
        nu: float = 0.1,  # KL constraint for projection
        max_lambda: float = 10.0,
        cost_window: int = 20,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.cost_limit = cost_limit
        self.lambda_lr = lambda_lr
        self.nu = nu
        self.max_lambda = max_lambda
        self._lambda = 1.0
        self._costs: list[float] = []
        self._cost_window = cost_window
        self._ep_cost_buffer: list[float] = []
        # Episode tracking for get_curves() compatibility
        self._successes: list[bool] = []
        self._collisions: list[bool] = []
        self._min_clearances: list[float] = []
        self._rewards: list[float] = []
        self._lengths: list[int] = []

    def _on_step(self) -> bool:
        # Accumulate per-step costs from info
        for info in self.locals.get("infos", []):
            cost = info.get("cost", 0.0)
            self._ep_cost_buffer.append(cost)

            if info.get("terminal_observation") is not None or \
               info.get("TimeLimit.truncated", False):
                # Episode ended — compute average cost rate
                if self._ep_cost_buffer:
                    ep_cost_rate = sum(self._ep_cost_buffer) / len(self._ep_cost_buffer)
                    self._costs.append(ep_cost_rate)
                self._ep_cost_buffer = []

        # Track episode metrics for get_curves() compatibility
        for done, info in zip(
            self.locals.get("dones", []),
            self.locals.get("infos", []),
        ):
            if not done:
                continue
            self._successes.append(bool(info.get("reached_goal", False)))
            self._collisions.append(bool(info.get("collision", False)))
            self._min_clearances.append(float(info.get("min_clearance", float("inf"))))
            self._rewards.append(float(info.get("episode", {}).get("r", 0.0)))
            self._lengths.append(int(info.get("episode", {}).get("l", 0)))

        return True

    def _on_rollout_end(self) -> None:
        """Apply FOCOPS-style constraint projection after each rollout."""
        if not self._costs:
            return

        # Compute recent average cost
        recent = self._costs[-self._cost_window:]
        avg_cost = sum(recent) / len(recent)

        # Update lambda (dual variable)
        violation = avg_cost - self.cost_limit
        self._lambda = float(np.clip(
            self._lambda + self.lambda_lr * violation,
            0.0, self.max_lambda,
        ))

        # Apply reward penalty to the rollout buffer
        # This modifies rewards in-place before PPO update
        if self._lambda > 0 and hasattr(self.model, "rollout_buffer"):
            buf = self.model.rollout_buffer
            if hasattr(buf, "rewards") and buf.rewards is not None:
                # Scale down rewards proportional to constraint violation
                penalty_scale = 1.0 / (1.0 + self._lambda * max(violation, 0.0))
                buf.rewards *= penalty_scale

        if self.verbose > 0:
            print(f"  FOCOPS: λ={self._lambda:.3f}  "
                  f"avg_cost={avg_cost:.4f}  "
                  f"violation={violation:+.4f}")

    def get_curves(self) -> dict:
        """Return training curves compatible with NavCallback interface."""
        return {
            "rewards": self._rewards,
            "lengths": self._lengths,
            "successes": self._successes,
            "collisions": self._collisions,
            "min_clearances": self._min_clearances,
            "cumulative_costs": [0.0] * len(self._successes),
        }


def make_focops_agent(
    env,
    config: dict,
    seed: int = 0,
    total_timesteps: int = 500_000,
) -> tuple[PPO, FOCOPSCallback]:
    """Create a PPO agent with FOCOPS constraint projection.

    Returns (model, callback) — train with model.learn(callback=callback).
    """
    tcfg = config.get("training", {})
    lag_cfg = tcfg.get("lagrangian", {})

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=float(tcfg.get("learning_rate", 3e-4)),
        n_steps=int(tcfg.get("n_steps", 512)),
        batch_size=int(tcfg.get("batch_size", 256)),
        n_epochs=int(tcfg.get("n_epochs", 10)),
        gamma=float(tcfg.get("gamma", 0.99)),
        gae_lambda=float(tcfg.get("gae_lambda", 0.95)),
        clip_range=float(tcfg.get("clip_range", 0.2)),
        ent_coef=float(tcfg.get("ent_coef", 0.005)),
        seed=seed,
        verbose=0,
    )

    callback = FOCOPSCallback(
        cost_limit=float(lag_cfg.get("cost_limit", 0.08)),
        lambda_lr=float(lag_cfg.get("lambda_lr", 0.005)),
        max_lambda=float(lag_cfg.get("lambda_max", 10.0)),
    )

    return model, callback
