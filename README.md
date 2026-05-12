# DAARS: Distance-Aware Adaptive Reward Shaping for Safe Robot Navigation

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-green.svg)](https://stable-baselines3.readthedocs.io/)

---

## Overview

DAARS replaces fixed reward weights in safe navigation with **distance-dependent scaling functions** that dynamically modulate progress and safety components based on obstacle proximity. Near obstacles, progress drive is suppressed while safety penalty is amplified; in open space, the reward degenerates to standard fixed-weight shaping.

---

## Noted timing on (8-core CPU, 1660ti)

| Phase | Wall time | Command |
|---|---|---|
| Training (all 6 methods × 5 seeds) | **~70 min** | `--phase train` |
| Evaluation (100 ep × 3 scenarios) | **~20 min** | `--phase evaluate` |
| Ablation (5×5 grid) | **~35 min** | `--phase ablation` |
| Theory + Plots | **~5 min** | `--phase theory && --phase plot` |
| **Total** | **~2.5 h** | `(full pipeline)` |

Previous timing was 2.8 h/run × 60 runs = 167 h.
Now: **~7 min/run × 30 runs = 3.5 h** via:
1. `SubprocVecEnv` (8 parallel env workers → 7× throughput)
2. `ProcessPoolExecutor` (3 methods run simultaneously)
3. 300k timesteps × 8 envs = 2.4M effective env-steps (same diversity as 1M single-env)
4. 5 seeds (sufficient for Wilcoxon signed-rank; standard in safe-RL literature)

---

## Quick start

```bash
# 1. Install
conda create -n daars python=3.10 && conda activate daars
pip install -r requirements.txt

# 2. Smoke test (takes ~3 min, confirms everything works)
python -m daars.run_experiments --fast --output results_test/

# 3. Full pipeline
python -m daars.run_experiments --output results/

# 4. Or phase by phase (recommended — easier to monitor)
python -m daars.run_experiments --phase train    --output results/
python -m daars.run_experiments --phase evaluate --output results/
python -m daars.run_experiments --phase ablation --output results/
python -m daars.run_experiments --phase theory   --output results/
python -m daars.run_experiments --phase plot     --output results/
```

## Tuning parallelism

```bash

# Example: 16-core machine — 4 parallel jobs × 4 envs each = 16 cores
python -m daars.run_experiments --phase train --workers 4 --envs 4

# Low RAM / 4-core machine
python -m daars.run_experiments --phase train --workers 1 --envs 4
```

## If a run crashes mid-way

The pipeline saves results after **every completed run**.
We just need to re-run the same command — it skips any models that already exist.

```bash
python -m daars.run_experiments --phase train --output results/
# will skip completed seeds and continue from where it left off
```

---

## Repository structure

```
daars/
├── config/default.yaml     ← ALL hyperparameters — edit here
├── rewards/
│   ├── daars_reward.py     ← Core contribution (α/β functions + proofs)
│   └── baselines.py        ← Static, α-only, β-only, task reward
├── envs/nav_env.py         ← 2-D sim with domain randomisation
├── agents/lagrangian.py    ← PPO-Lag + SAC-Lag (proper baselines)
├── training/train.py       ← SubprocVecEnv + parallel launcher
├── evaluation/evaluate.py  ← Fixed path-efficiency metric
├── analysis/
│   ├── statistics.py       ← Wilcoxon, bootstrap CI, Cohen's d
│   └── plots.py            ← 7  figures (PDF + PNG)
├── theory/analysis.py      ← Formal property verification P1–P4
├── gazebo/gazebo_env.py    ← ROS2/Gazebo drop-in bridge
└── run_experiments.py      ← Main pipeline
scripts/
└── run_gazebo.py           ← Deploy trained policy in Gazebo
```

---

## Methods compared

| ID | Type | Description |
|---|---|---|
| `static` | Baseline | Fixed-weight reward shaping + PPO |
| `alpha_only` | Ablation | Progress suppression only |
| `beta_only` | Ablation | Safety amplification only |
| `daars` | **Proposed** | Full α+β dual modulation + PPO |
| `ppo_lag` | Baseline | PPO-Lagrangian (constrained RL) |
| `sac_lag` | Baseline | SAC-Lagrangian (sample-efficient constrained RL) |

---

## Gazebo validation (for real-robot section)

```bash
# Terminal 1
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# Terminal 2
source /opt/ros/humble/setup.bash
python scripts/run_gazebo.py \
    --model results/models/daars_seed0 \
    --episodes 50 --output gazebo_results.json
```