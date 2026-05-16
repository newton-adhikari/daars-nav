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

source /opt/ros/humble/setup.bash
source "$REPO/install/setup.bash" 2>/dev/null || true
export TURTLEBOT3_MODEL=burger
# Headless — gazebo (saves CPU)
export GAZEBO_MODEL_PATH="/opt/ros/humble/share/turtlebot3_gazebo/models:${GAZEBO_MODEL_PATH:-}"

WORLD_FILE="$REPO/install/daars_gazebo/share/daars_gazebo/worlds/daars_arena.world"
if [ ! -f "$WORLD_FILE" ]; then
    WORLD_FILE="$REPO/daars_gazebo/worlds/daars_arena.world"
fi
TB3_SDF="/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf"
TB3_URDF="/opt/ros/humble/share/turtlebot3_description/urdf/turtlebot3_burger.urdf"

log() { echo "[$(date +%H:%M:%S)] $*"; }

cleanup() {
    log "Cleaning up..."
    pkill -f gzserver 2>/dev/null || true
    pkill -f robot_state_publisher 2>/dev/null || true
    sleep 2
    pkill -9 -f gzserver 2>/dev/null || true
    pkill -9 -f robot_state_publisher 2>/dev/null || true
}
trap cleanup EXIT

# Kill any leftover processes
cleanup
sleep 2

echo "========================================"
echo "  DAARS Gazebo Evaluation (headless)"
echo "  Methods: ${METHODS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "  Episodes/seed: ${EPISODES_PER_SEED}"
echo "========================================"
echo ""

# Start Gazebo ONCE (headless — no gzclient)
log "Starting Gazebo (headless)..."
gzserver "$WORLD_FILE" \
    -s libgazebo_ros_init.so \
    -s libgazebo_ros_factory.so \
    -s libgazebo_ros_force_system.so &
GZPID=$!

# Wait for spawn service
log "Waiting for Gazebo services..."
for i in $(seq 1 60); do
    if ros2 service list 2>/dev/null | grep -q "/spawn_entity"; then
        log "Gazebo ready (${i}s)"
        break
    fi
    if [ $i -eq 60 ]; then
        log "ERROR: Gazebo failed to start after 60s"
        exit 1
    fi
    sleep 1
done

# Start robot state publisher
ros2 run robot_state_publisher robot_state_publisher \
    --ros-args -p robot_description:="$(cat $TB3_URDF)" &

# Initial spawn
ros2 run gazebo_ros spawn_entity.py \
    -entity burger -file "$TB3_SDF" \
    -x 1.0 -y 1.0 -z 0.01 > /dev/null 2>&1

# Wait for /scan topic
for i in $(seq 1 20); do
    if ros2 topic list 2>/dev/null | grep -q "/scan"; then
        log "LiDAR active"
        break
    fi
    sleep 1
done
sleep 2

# Run all methods and seeds in a single Gazebo session
TOTAL=0
DONE=0
for METHOD in "${METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
    done
done

log "Running $TOTAL evaluation jobs..."
echo ""

for METHOD in "${METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        MODEL_PATH="${MODEL_DIR}/${METHOD}_seed${SEED}"
        SEED_OUTPUT="${OUTPUT_DIR}/${METHOD}_seed${SEED}.json"

        DONE=$((DONE + 1))

        if [ ! -f "${MODEL_PATH}.zip" ]; then
            log "[$DONE/$TOTAL] SKIP ${METHOD}/seed${SEED} (no model)"
            continue
        fi

        if [ -f "$SEED_OUTPUT" ]; then
            EP_COUNT=$(python3 -c "import json; print(len(json.load(open('$SEED_OUTPUT'))))" 2>/dev/null || echo 0)
            if [ "$EP_COUNT" -ge "$EPISODES_PER_SEED" ]; then
                log "[$DONE/$TOTAL] SKIP ${METHOD}/seed${SEED} (done)"
                continue
            fi
        fi

        log "[$DONE/$TOTAL] ${METHOD}/seed${SEED} (${EPISODES_PER_SEED} eps)..."
        python3 "$REPO/scripts/run_gazebo.py" \
            --model "$MODEL_PATH" \
            --episodes "$EPISODES_PER_SEED" \
            --output "$SEED_OUTPUT" \
            || log "  WARNING: ${METHOD}/seed${SEED} had errors"
    done
done

echo ""
log "All done! analyze results::: "
 