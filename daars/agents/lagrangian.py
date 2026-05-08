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


# try to create shared Lagrangian multiplier state
# TODO: may need to update later


class LagrangianMultiplier:

    def __init__(
        self,
        init_value: float = 1.0,
        lr: float = 0.01,
        cost_limit: float = 0.05,
        max_value: float = 20.0,
        window: int = 20,
    ):
        self.value      = init_value
        self.lr         = lr
        self.cost_limit = cost_limit
        self.max_value  = max_value
        self._costs: list[float] = []
        self._window    = window

    def update(self, episode_avg_cost: float) -> None:
        """Dual ascent step.

        Args:
            episode_avg_cost: mean cost per step for the latest episode
                              (i.e. cumulative_cost / episode_length).
        """
        self._costs.append(episode_avg_cost)
        recent = self._costs[-self._window:]
        avg    = sum(recent) / len(recent)
        self.value = float(np.clip(
            self.value + self.lr * (avg - self.cost_limit),
            0.0, self.max_value,
        ))

    def get(self) -> float:
        return self.value

    def reset(self) -> None:
        self.value  = 1.0
        self._costs = []


# defined a callback to, injects Lagrangian penalty into rewards and updates λ

class LagrangianCallback(BaseCallback):
    """
    SB3 callback that implements the Lagrangian reward adjustment.
    """

    def __init__(
        self,
        lagrangian: LagrangianMultiplier,
        reward_type: str = "lag",
        total_timesteps: int = 300_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.lagrangian      = lagrangian
        self.reward_type     = reward_type
        self.total_timesteps = total_timesteps
        self.episode_rewards : list[float] = []
        self.episode_lengths : list[int]   = []
        self.successes       : list[bool]  = []
        self.collisions      : list[bool]  = []
        self.min_clearances  : list[float] = []
        self.lambda_history  : list[float] = []
        self._ep_reward      = 0.0
        self._ep_len         = 0
        self._start_time     = None
        self._last_log       = 0
        self._log_interval   = 10_000

    def _on_training_start(self):
        import time
        self._start_time = time.time()

    def _on_step(self) -> bool:
        import sys, time
        self._ep_reward += float(self.locals["rewards"][0])
        self._ep_len    += 1

        # Inject Lagrangian penalty: r_modified = r - λ * cost
        # This is the core mechanism that makes Lagrangian methods work.
        infos = self.locals.get("infos", [{}])
        lam = self.lagrangian.get()
        for i, info in enumerate(infos):
            cost = float(info.get("cost", 0.0))
            if cost > 0 and lam > 0:
                self.locals["rewards"][i] -= lam * cost

        # Live progress (same style as NavCallback)
        if self.num_timesteps - self._last_log >= self._log_interval:
            self._last_log = self.num_timesteps
            elapsed = time.time() - (self._start_time or time.time())
            fps     = self.num_timesteps / max(elapsed, 1e-6)
            pct     = 100.0 * self.num_timesteps / self.total_timesteps
            rem     = (self.total_timesteps - self.num_timesteps) / max(fps, 1)
            n       = len(self.successes)
            sr = float(np.mean(self.successes[-50:]))  if n >= 50 else (float(np.mean(self.successes)) if n else 0)
            cr = float(np.mean(self.collisions[-50:])) if n >= 50 else (float(np.mean(self.collisions)) if n else 0)
            sys.stdout.write(
                f"\r  [{self.reward_type:10s}] {pct:5.1f}%  "
                f"{fps:6.0f} fps  ~{rem:5.0f}s  "
                f"ep={n:4d}  SR={sr:.3f}  CR={cr:.3f}  λ={self.lagrangian.get():.2f}    "
            )
            sys.stdout.flush()

        dones = self.locals.get("dones", [False])
        infos = self.locals.get("infos", [{}])

        for done, info in zip(dones, infos):
            if not done:
                continue
            # Episode stats
            self.episode_rewards.append(self._ep_reward)
            self.episode_lengths.append(self._ep_len)
            self.successes.append(bool(info.get("reached_goal", False)))
            self.collisions.append(bool(info.get("collision", False)))
            self.min_clearances.append(float(info.get("min_clearance", float("inf"))))

            # Update λ using average cost per step this episode
            cum_cost  = float(info.get("cumulative_cost", 0.0))
            ep_length = max(self._ep_len, 1)
            self.lagrangian.update(cum_cost / ep_length)
            self.lambda_history.append(self.lagrangian.get())

            self._ep_reward = 0.0
            self._ep_len    = 0

        return True

    def get_curves(self) -> dict:
        return {
            "rewards":       self.episode_rewards,
            "lengths":       self.episode_lengths,
            "successes":     self.successes,
            "collisions":    self.collisions,
            "min_clearances": self.min_clearances,
            "lambda":        self.lambda_history,
        }


# PPO-Lagrangian

def make_ppo_lagrangian(
    env,
    config: dict,
    lagrangian: LagrangianMultiplier,
    seed: int = 0,
) -> tuple[PPO, LagrangianCallback]:
    """
    Construct a PPO model and with Lagrangian callback.
    """
    tcfg = config["training"]

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate  = tcfg["learning_rate"],
        n_steps        = tcfg["n_steps"],
        batch_size     = tcfg["batch_size"],
        n_epochs       = tcfg["n_epochs"],
        gamma          = tcfg["gamma"],
        gae_lambda     = tcfg["gae_lambda"],
        clip_range     = tcfg["clip_range"],
        ent_coef       = tcfg["ent_coef"],
        vf_coef        = tcfg["vf_coef"],
        max_grad_norm  = tcfg["max_grad_norm"],
        policy_kwargs  = dict(net_arch=tcfg["policy_layers"]),
        verbose        = 0,
        seed           = seed,
    )

    callback = LagrangianCallback(lagrangian, reward_type="ppo_lag",
                                   total_timesteps=int(tcfg["total_timesteps"]))
    return model, callback


# SAC-Lagrangian

def make_sac_lagrangian(
    env,
    config: dict,
    lagrangian: LagrangianMultiplier,
    seed: int = 0,
) -> tuple[SAC, LagrangianCallback]:
    """
    Construct a SAC model for the Lagrangian constrained baseline.
    """
    tcfg = config["training"]

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate  = tcfg["learning_rate"],
        gamma          = tcfg["gamma"],
        batch_size     = tcfg["batch_size"],
        train_freq     = 8,            
        gradient_steps = 1,            # 1 gradient step per update
        learning_starts = 1000,        # warm up replay buffer
        buffer_size    = 100_000,      
        policy_kwargs  = dict(net_arch=tcfg["policy_layers"]),
        verbose        = 0,
        seed           = seed,
    )

    callback = LagrangianCallback(lagrangian, reward_type="sac_lag",
                                   total_timesteps=int(tcfg["total_timesteps"]))
    return model, callback
 