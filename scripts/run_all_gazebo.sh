#!/usr/bin/env bash
# Gazebo evaluation —  headless mode.

set -e

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPISODES_PER_SEED=10
SEEDS=(0 1 2 3 4)
METHODS=("daars" "static" "alpha_only" "beta_only" "ppo_lag")
OUTPUT_DIR="$REPO/results/gazebo"
MODEL_DIR="$REPO/results/models"

mkdir -p "$OUTPUT_DIR"
