#!/usr/bin/env bash
#
# DAARS Gazebo Evaluation Pipeline (Headless)
#
# Runs all methods × seeds in a single Gazebo session.
# Produces results/gazebo/*.json files for analysis.
#
# Prerequisites:
#   - ROS 2 Humble installed
#   - TurtleBot3 packages installed
#   - Gazebo classic (gazebo11) installed
#   - Trained models in results/models/
#
# Usage:
#   ./scripts/run_all_gazebo.sh
#   ./scripts/run_all_gazebo.sh 2>&1 | tee gazebo_run.log

set -e

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ─── Configuration ───────────────────────────────────────────────────────
EPISODES_PER_SEED=20
SEEDS=(0 1 2 3 4)
METHODS=("daars" "static" "alpha_only" "beta_only" "ppo_lag" "focops")
OUTPUT_DIR="$REPO/results/gazebo"
MODEL_DIR="$REPO/results/models"
LOG_FILE="$OUTPUT_DIR/gazebo_eval.log"
# ─────────────────────────────────────────────────────────────────────────

mkdir -p "$OUTPUT_DIR"

# Source ROS 2
source /opt/ros/humble/setup.bash
source "$REPO/install/setup.bash" 2>/dev/null || true
export TURTLEBOT3_MODEL=burger
export GAZEBO_MODEL_PATH="/opt/ros/humble/share/turtlebot3_gazebo/models:${GAZEBO_MODEL_PATH:-}"

# Find world file
WORLD_FILE="$REPO/daars_gazebo/worlds/daars_arena.world"
if [ ! -f "$WORLD_FILE" ]; then
    echo "ERROR: World file not found: $WORLD_FILE"
    exit 1
fi

TB3_SDF="/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf"
TB3_URDF="/opt/ros/humble/share/turtlebot3_description/urdf/turtlebot3_burger.urdf"

# ─── Logging ─────────────────────────────────────────────────────────────
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

# ─── Cleanup ─────────────────────────────────────────────────────────────
cleanup() {
    log "Shutting down Gazebo..."
    pkill -f gzserver 2>/dev/null || true
    pkill -f robot_state_publisher 2>/dev/null || true
    sleep 1
    pkill -9 -f gzserver 2>/dev/null || true
    pkill -9 -f robot_state_publisher 2>/dev/null || true
}
trap cleanup EXIT

# Kill leftover processes from previous runs
cleanup
sleep 2

# ─── Start ───────────────────────────────────────────────────────────────
echo "=== DAARS Gazebo Evaluation ===" > "$LOG_FILE"
log "Configuration:"
log "  Methods:  ${METHODS[*]}"
log "  Seeds:    ${SEEDS[*]}"
log "  Episodes: ${EPISODES_PER_SEED}/seed"
log "  Output:   ${OUTPUT_DIR}/"
log ""

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  DAARS Gazebo Evaluation (headless)"
echo "  Methods: ${METHODS[*]}"
echo "  Seeds:   ${SEEDS[*]}"
echo "  Eps/seed: ${EPISODES_PER_SEED}"
echo "  Output:  ${OUTPUT_DIR}/"
echo "════════════════════════════════════════════════════════════════"
echo ""

# ─── Launch Gazebo ───────────────────────────────────────────────────────
log "Starting Gazebo server (headless)..."
gzserver "$WORLD_FILE" \
    -s libgazebo_ros_init.so \
    -s libgazebo_ros_factory.so \
    --verbose &
GZPID=$!

# Wait for Gazebo services
log "Waiting for Gazebo to be ready..."
for i in $(seq 1 90); do
    if ros2 service list 2>/dev/null | grep -q "/spawn_entity"; then
        log "Gazebo ready (${i}s)"
        break
    fi
    if [ $i -eq 90 ]; then
        log "ERROR: Gazebo failed to start after 90s"
        exit 1
    fi
    sleep 1
done

# Robot state publisher
ros2 run robot_state_publisher robot_state_publisher \
    --ros-args -p robot_description:="$(cat $TB3_URDF)" &
sleep 1

# Initial robot spawn
log "Spawning TurtleBot3..."
ros2 run gazebo_ros spawn_entity.py \
    -entity burger -file "$TB3_SDF" \
    -x 1.0 -y 1.0 -z 0.01 > /dev/null 2>&1
sleep 2

# Verify LiDAR
for i in $(seq 1 30); do
    if ros2 topic list 2>/dev/null | grep -q "/scan"; then
        log "LiDAR topic active"
        break
    fi
    if [ $i -eq 30 ]; then
        log "WARNING: /scan topic not found after 30s"
    fi
    sleep 1
done
sleep 1

# ─── Run Evaluations ────────────────────────────────────────────────────
TOTAL=0
for METHOD in "${METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
    done
done

log ""
log "Starting $TOTAL evaluation jobs..."
echo ""

DONE=0
SKIPPED=0
FAILED=0
START_TIME=$(date +%s)

for METHOD in "${METHODS[@]}"; do
    log "── Method: ${METHOD} ──"

    for SEED in "${SEEDS[@]}"; do
        MODEL_PATH="${MODEL_DIR}/${METHOD}_seed${SEED}"
        SEED_OUTPUT="${OUTPUT_DIR}/${METHOD}_seed${SEED}.json"
        DONE=$((DONE + 1))

        # Skip if no model
        if [ ! -f "${MODEL_PATH}.zip" ]; then
            log "  [$DONE/$TOTAL] SKIP ${METHOD}/seed${SEED} — no model"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        # Skip if already completed
        if [ -f "$SEED_OUTPUT" ]; then
            EP_COUNT=$(python3 -c "
import json
try:
    d = json.load(open('$SEED_OUTPUT'))
    print(len(d.get('episodes', d if isinstance(d, list) else [])))
except: print(0)
" 2>/dev/null)
            if [ "$EP_COUNT" -ge "$EPISODES_PER_SEED" ]; then
                log "  [$DONE/$TOTAL] SKIP ${METHOD}/seed${SEED} — done (${EP_COUNT} eps)"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
        fi

        # Run evaluation
        log "  [$DONE/$TOTAL] RUN ${METHOD}/seed${SEED} (${EPISODES_PER_SEED} eps)..."
        JOB_START=$(date +%s)

        if python3 "$REPO/scripts/run_gazebo.py" \
            --model "$MODEL_PATH" \
            --method "$METHOD" \
            --episodes "$EPISODES_PER_SEED" \
            --output "$SEED_OUTPUT" \
            2>&1 | tee -a "$LOG_FILE"; then

            JOB_END=$(date +%s)
            JOB_TIME=$((JOB_END - JOB_START))
            ELAPSED=$((JOB_END - START_TIME))
            JOBS_RUN=$((DONE - SKIPPED - FAILED))
            if [ $JOBS_RUN -gt 0 ]; then
                AVG=$((ELAPSED / JOBS_RUN))
                REMAINING=$((TOTAL - DONE))
                ETA=$((AVG * REMAINING))
            else
                ETA=0
            fi
            log "  ✓ Done in ${JOB_TIME}s | Total elapsed: ${ELAPSED}s | ETA: ~${ETA}s"
        else
            FAILED=$((FAILED + 1))
            log "  ✗ FAILED ${METHOD}/seed${SEED}"
        fi
        echo ""
    done
done

# ─── Summary ─────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
JOBS_RUN=$((DONE - SKIPPED - FAILED))

echo ""
log "════════════════════════════════════════════════════════════════"
log "  EVALUATION COMPLETE"
log "  Total time:  ${TOTAL_TIME}s ($(( TOTAL_TIME / 60 ))m)"
log "  Jobs run:    ${JOBS_RUN}"
log "  Skipped:     ${SKIPPED}"
log "  Failed:      ${FAILED}"
log "  Results:     ${OUTPUT_DIR}/"
log "  Log:         ${LOG_FILE}"
log "════════════════════════════════════════════════════════════════"
log ""
log "Next step: python3 scripts/analyze_gazebo.py"
 