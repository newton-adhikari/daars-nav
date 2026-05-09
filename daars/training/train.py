# this is a script to run the training with daars rewards

from __future__ import annotations

import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from daars.envs import RobotNavEnv
from daars.rewards import REWARD_FUNCTIONS, COST_FUNCTIONS
from daars.agents.lagrangian import (
    LagrangianMultiplier, LagrangianCallback,
    make_ppo_lagrangian, make_sac_lagrangian,
)
from daars.agents.hybrid import make_daars_lagrangian
from daars.training.callbacks import NavCallback



# Env factories

def _env_fn(config, reward_fn, cost_fn, scenario, seed):
    def _init():
        return RobotNavEnv(config, reward_fn=reward_fn, cost_fn=cost_fn,
                           scenario=scenario, seed=seed, domain_rand=True)
    return _init


def _make_vec(config, reward_fn, cost_fn, scenario, seed, num_envs):
    """Building SubprocVecEnv .
    """
    fns = [_env_fn(config, reward_fn, cost_fn, scenario, seed * 1000 + i)
           for i in range(num_envs)]
    if num_envs == 1:
        return DummyVecEnv(fns)

    try:
        return SubprocVecEnv(fns, start_method="fork")
    except Exception:
        return SubprocVecEnv(fns, start_method="spawn")



# Single-run training
def train_agent(
    config: dict,
    reward_type: str,
    seed: int,
    scenario: str = "simple",
    save_dir: str = "models",
    daars_ks: float | None = None,
    daars_dsafe: float | None = None,
    num_envs: int | None = None,
) -> dict:
    """Training one agent, return training curves + metadata."""
    tcfg     = config["training"]
    total_ts = int(tcfg["total_timesteps"])
    n_envs   = num_envs if num_envs is not None else int(tcfg.get("num_envs", 8))
    is_sac   = reward_type == "sac_lag"
    is_lag   = reward_type in ("ppo_lag", "sac_lag", "daars_lag")

    # it appears SAC does not support VecEnv — so we fall back to single env.
    if is_sac:
        n_envs   = 1

    print(f"\n>> {reward_type:12s}  seed={seed}  "
          f"envs={n_envs}  steps={total_ts:,}  ({scenario})")

    reward_fn = REWARD_FUNCTIONS[reward_type]
    cost_fn   = COST_FUNCTIONS[reward_type]

    # Learned DAARS: create modulator and wrap reward function
    modulator_mgr = None
    if reward_type == "learned":
        from daars.rewards.learned_daars import LearnedModulatorManager, learned_daars_reward
        modulator_mgr = LearnedModulatorManager(lr=1e-3, update_every=50)
        _mod = modulator_mgr
        reward_fn = lambda info, cfg, __m=_mod: learned_daars_reward(info, cfg, modulator=__m)
        # Use DummyVecEnv for learned (modulator must be in same process)
        n_envs = min(n_envs, 4)

    if reward_type == "daars" and (daars_ks is not None or daars_dsafe is not None):
        _base, _ks, _ds = reward_fn, daars_ks, daars_dsafe
        reward_fn = lambda i, c, __b=_base, __ks=_ks, __ds=_ds: __b(i, c, ks=__ks, dsafe=__ds)

    env = _make_vec(config, reward_fn, cost_fn, scenario, seed, n_envs)

    if is_lag:
        lag_cfg    = tcfg.get("lagrangian", {})
        lagrangian = LagrangianMultiplier(
            init_value = float(lag_cfg.get("init_lambda", 1.0)),
            lr         = float(lag_cfg.get("lambda_lr",  0.01)),
            cost_limit = float(lag_cfg.get("cost_limit", 0.05)),
            max_value  = float(lag_cfg.get("lambda_max", 20.0)),
        )
        if is_sac:
            model, callback = make_sac_lagrangian(env, config, lagrangian, seed)
        elif reward_type == "daars_lag":
            model, callback = make_daars_lagrangian(env, config, lagrangian, seed)
        else:
            model, callback = make_ppo_lagrangian(env, config, lagrangian, seed)
    elif reward_type == "learned":
        from daars.training.learned_callback import LearnedDAARSCallback
        callback = LearnedDAARSCallback(modulator_mgr, reward_type="learned",
                                         total_timesteps=total_ts)
        model = PPO(
            "MlpPolicy", env,
            learning_rate = float(tcfg["learning_rate"]),
            n_steps       = int(tcfg["n_steps"]),
            batch_size    = int(tcfg["batch_size"]),
            n_epochs      = int(tcfg["n_epochs"]),
            gamma         = float(tcfg["gamma"]),
            gae_lambda    = float(tcfg["gae_lambda"]),
            clip_range    = float(tcfg["clip_range"]),
            ent_coef      = float(tcfg["ent_coef"]),
            vf_coef       = float(tcfg["vf_coef"]),
            max_grad_norm = float(tcfg["max_grad_norm"]),
            policy_kwargs = dict(net_arch=list(tcfg["policy_layers"])),
            verbose=0, seed=seed,
        )
    else:
        callback = NavCallback(reward_type=reward_type, total_timesteps=total_ts)
        model    = PPO(
            "MlpPolicy", env,
            learning_rate = float(tcfg["learning_rate"]),
            n_steps       = int(tcfg["n_steps"]),
            batch_size    = int(tcfg["batch_size"]),
            n_epochs      = int(tcfg["n_epochs"]),
            gamma         = float(tcfg["gamma"]),
            gae_lambda    = float(tcfg["gae_lambda"]),
            clip_range    = float(tcfg["clip_range"]),
            ent_coef      = float(tcfg["ent_coef"]),
            vf_coef       = float(tcfg["vf_coef"]),
            max_grad_norm = float(tcfg["max_grad_norm"]),
            policy_kwargs = dict(net_arch=list(tcfg["policy_layers"])),
            verbose=0, seed=seed,
        )

    t0 = time.time()
    model.learn(total_timesteps=total_ts, callback=callback)
    elapsed = time.time() - t0

    curves = callback.get_curves()
    n_ep   = len(curves.get("successes", []))
    sr = float(np.mean(curves["successes"][-50:])) if n_ep >= 50 else \
         (float(np.mean(curves["successes"])) if n_ep else 0.0)
    cr = float(np.mean(curves["collisions"][-50:])) if n_ep >= 50 else \
         (float(np.mean(curves["collisions"])) if n_ep else 0.0)
    print(f"   ✓ {elapsed/60:.1f} min | {n_ep} eps | SR={sr:.3f} CR={cr:.3f}")

    os.makedirs(save_dir, exist_ok=True)
    suffix = ""
    if daars_ks    is not None: suffix += f"_ks{daars_ks}"
    if daars_dsafe is not None: suffix += f"_ds{daars_dsafe}"
    path = os.path.join(save_dir, f"{reward_type}{suffix}_seed{seed}")
    model.save(path)
    env.close()

    return {"model_path": path, "training_curves": curves,
            "reward_type": reward_type, "seed": seed}



# Parallel seed launcher

def _worker(args):
    config, rtype, seed, scenario, save_dir, ks, ds, n_envs = args
    try:
        return train_agent(config, rtype, seed, scenario, save_dir,
                           daars_ks=ks, daars_dsafe=ds, num_envs=n_envs)
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "reward_type": rtype, "seed": seed}


def train_all_parallel(
    config: dict,
    methods: list[str],
    save_dir: str,
    scenario: str = "simple",
    max_workers: int | None = None,
) -> dict[str, list]:
    """ created this to run all method × seed combinations in parallel.

    also the jobs that already have a saved model are skipped automatically.
    """
    tcfg   = config["training"]
    seeds  = list(range(int(tcfg["num_seeds"])))
    n_envs = int(tcfg.get("num_envs", 8))
    max_w  = max_workers or int(tcfg.get("max_parallel_jobs", 3))

    jobs = []
    for method in methods:
        for seed in seeds:
            if os.path.exists(os.path.join(save_dir, f"{method}_seed{seed}.zip")):
                print(f"   skip {method}/seed{seed}")
                continue
            jobs.append((config, method, seed, scenario,
                         save_dir, None, None, n_envs))

    if not jobs:
        print("   All models already trained.")
        return {m: [] for m in methods}

    effective_w = max_w

    print(f"\n>> Parallel launcher: {len(jobs)} jobs  "
          f"max_workers={effective_w}  envs/job={n_envs}")
    sac_count = sum(1 for j in jobs if j[1] == "sac_lag")
    if sac_count:
        print(f"   ({sac_count} SAC jobs @ 300k steps, train_freq=8)")
    print(f"   Estimated time: ~{len(jobs)/effective_w * 7:.0f} min\n")

    results: dict[str, list] = {m: [] for m in methods}
    import multiprocessing as mp
    ctx = mp.get_context("spawn")          # BUG FIX: fork + PyTorch = deadlock
    with ProcessPoolExecutor(max_workers=effective_w, mp_context=ctx) as pool:
        futures = {pool.submit(_worker, j): j for j in jobs}
        for fut in as_completed(futures):
            res = fut.result()
            if "error" not in res:
                results[res["reward_type"]].append(res)
            else:
                j = futures[fut]
                print(f"!! FAILED {j[1]}/seed{j[2]}: {res['error']}")
    return results
 