"""
DAARS + Lagrangian Hybrid Agent.
"""

from __future__ import annotations

from stable_baselines3 import PPO
from daars.agents.lagrangian import LagrangianMultiplier, LagrangianCallback


def make_daars_lagrangian(
    env,
    config: dict,
    lagrangian: LagrangianMultiplier,
    seed: int = 0,
) -> tuple[PPO, LagrangianCallback]:
    """
    PPO with DAARS-shaped reward AND Lagrangian cost constraint.
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

    callback = LagrangianCallback(lagrangian, reward_type="daars_lag",
                                   total_timesteps=int(tcfg["total_timesteps"]))
    return model, callback
