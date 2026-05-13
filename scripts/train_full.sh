#!/usr/bin/env bash
# Full retraining pipeline for daars.

set -e

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${1:-all}"

echo "========================================"
echo "  DAARS Full Retraining Pipeline"
echo "  Methods: static, alpha_only, daars, beta_only, ppo_lag"
echo "  Seeds: 20"
echo "  Timesteps: 500k (8 envs)"
echo "  Ablation: 25 cells × 5 seeds"
echo "========================================"
echo ""
echo "[$(date)] Starting phase: ${PHASE}"
echo ""

cd "$REPO"
python3 -m daars --phase "$PHASE" --output results

echo ""
echo "[$(date)] Phase '${PHASE}' complete."
 