#!/usr/bin/env bash
# Full retraining pipeline for DAARS (revised for ICRA/IROS resubmission).

set -e

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${1:-all}"

echo "========================================"
echo "  DAARS Full Retraining Pipeline (REVISED)"
echo "  Methods: static, alpha_only, daars, beta_only, ppo_lag, focops"
echo "  Seeds: 20"
echo "  Timesteps: 1M (extended for convergence)"
echo "  Ablation: 25 cells × 5 seeds"
echo "  Safety-Gymnasium: 3 envs × 4 methods × 10 seeds"
echo "========================================"
echo ""
echo "[$(date)] Starting phase: ${PHASE}"
echo ""

cd "$REPO"

if [ "$PHASE" = "all" ]; then
    echo ">> Phase 1/6: Training (1M steps, 20 seeds, 6 methods)"
    python3 -m daars --phase train --extended --output results

    echo ""
    echo ">> Phase 2/6: Evaluation"
    python3 -m daars --phase evaluate --output results

    echo ""
    echo ">> Phase 3/6: Safety-Gymnasium benchmark"
    python3 -m daars --phase safetygym --output results

    echo ""
    echo ">> Phase 4/6: Ablation"
    python3 -m daars --phase ablation --output results

    echo ""
    echo ">> Phase 5/6: Theory verification"
    python3 -m daars --phase theory --output results

    echo ""
    echo ">> Phase 6/6: Statistics + plots"
    python3 -m daars --phase plot --output results

    echo ""
    echo ">> Generating corrected table"
    python3 scripts/generate_corrected_table.py > results/corrected_table.txt
else
    python3 -m daars --phase "$PHASE" --extended --output results
fi

echo ""
echo "[$(date)] Phase '${PHASE}' complete."
echo ""
echo "Results in: results/"
echo "  - statistics.json (with PE tests, convergence, variance analysis)"
echo "  - safety_gym_results.json"
echo "  - corrected_table.txt"
