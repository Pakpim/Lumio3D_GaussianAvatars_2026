#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/rpool/data/icy/venv/bin/python}"
SOURCE_PATH="${SOURCE_PATH:-/rpool/data/icy/dataset/bird_v01}"
RUN_NAME="${RUN_NAME:-fastgs_shortersplat_30000}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/output/$RUN_NAME}"
FASTGS_FILTER_INTERVAL="${FASTGS_FILTER_INTERVAL:-4}"
START_CHECKPOINT="${START_CHECKPOINT:-}"

# Densification schedule (speed-first defaults)
DENSIFICATION_INTERVAL="${DENSIFICATION_INTERVAL:-2000}"
OPACITY_RESET_INTERVAL="${OPACITY_RESET_INTERVAL:-5000}"
DENSIFY_FROM_ITER="${DENSIFY_FROM_ITER:-5000}"
DENSIFY_UNTIL_ITER="${DENSIFY_UNTIL_ITER:-20000}"

# ShorterGS controls
LAMBDA_ENTROPY="${LAMBDA_ENTROPY:-0.015}"
# Keep scale reset off by default for mesh-avatar stability; set >0 to enable.
SCALE_RESET_FACTOR="${SCALE_RESET_FACTOR:-0.0}"
SHORTER_SCALE_PRUNE_QUANTILE="${SHORTER_SCALE_PRUNE_QUANTILE:-0.99}"
SHORTER_SCALE_PRUNE_FACTOR="${SHORTER_SCALE_PRUNE_FACTOR:-10.0}"
SHORTER_PRUNE_MAX_FRACTION="${SHORTER_PRUNE_MAX_FRACTION:-0.05}"

# Speed guards for point growth (0/1, false/true, no/yes)
SHORTER_CLONE_INVISIBLE_POINTS="${SHORTER_CLONE_INVISIBLE_POINTS:-0}"
SHORTER_AUTO_TUNE_SCHEDULE="${SHORTER_AUTO_TUNE_SCHEDULE:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] Python executable not found: $PYTHON_BIN"
    exit 1
fi

mkdir -p "$MODEL_PATH"

echo "[INFO] Starting FastGS + ShorterSplat training"
echo "[INFO] source_path: $SOURCE_PATH"
echo "[INFO] model_path:  $MODEL_PATH"
echo "[INFO] fastgs_filter_interval: $FASTGS_FILTER_INTERVAL"
echo "[INFO] densification_interval: $DENSIFICATION_INTERVAL"
echo "[INFO] opacity_reset_interval: $OPACITY_RESET_INTERVAL"
echo "[INFO] densify_from_iter: $DENSIFY_FROM_ITER"
echo "[INFO] densify_until_iter: $DENSIFY_UNTIL_ITER"
echo "[INFO] lambda_entropy: $LAMBDA_ENTROPY"
echo "[INFO] scale_reset_factor: $SCALE_RESET_FACTOR"
echo "[INFO] shorter_scale_prune_quantile: $SHORTER_SCALE_PRUNE_QUANTILE"
echo "[INFO] shorter_scale_prune_factor: $SHORTER_SCALE_PRUNE_FACTOR"
echo "[INFO] shorter_prune_max_fraction: $SHORTER_PRUNE_MAX_FRACTION"
echo "[INFO] shorter_clone_invisible_points: $SHORTER_CLONE_INVISIBLE_POINTS"
echo "[INFO] shorter_auto_tune_schedule: $SHORTER_AUTO_TUNE_SCHEDULE"
if [[ -n "$START_CHECKPOINT" ]]; then
    echo "[INFO] start_checkpoint: $START_CHECKPOINT"
fi

if (( DENSIFY_UNTIL_ITER <= DENSIFY_FROM_ITER )); then
    echo "[ERROR] Invalid densify window: DENSIFY_UNTIL_ITER ($DENSIFY_UNTIL_ITER) must be > DENSIFY_FROM_ITER ($DENSIFY_FROM_ITER)."
    exit 1
fi

first_densify_iter=$(( (DENSIFY_FROM_ITER / DENSIFICATION_INTERVAL + 1) * DENSIFICATION_INTERVAL ))
if (( first_densify_iter >= DENSIFY_UNTIL_ITER )); then
    echo "[WARN] Current schedule yields zero densify/prune events in-window."
    echo "[WARN] first_densify_iter=$first_densify_iter, window=($DENSIFY_FROM_ITER,$DENSIFY_UNTIL_ITER), interval=$DENSIFICATION_INTERVAL"
    echo "[WARN] Suggested: lower DENSIFICATION_INTERVAL or increase DENSIFY_UNTIL_ITER."
fi

EXTRA_SHORTER_FLAGS=()
if [[ "$SHORTER_CLONE_INVISIBLE_POINTS" == "1" || "$SHORTER_CLONE_INVISIBLE_POINTS" == "true" || "$SHORTER_CLONE_INVISIBLE_POINTS" == "yes" ]]; then
    EXTRA_SHORTER_FLAGS+=(--shorter_clone_invisible_points)
fi
if [[ "$SHORTER_AUTO_TUNE_SCHEDULE" == "1" || "$SHORTER_AUTO_TUNE_SCHEDULE" == "true" || "$SHORTER_AUTO_TUNE_SCHEDULE" == "yes" ]]; then
    EXTRA_SHORTER_FLAGS+=(--shorter_auto_tune_schedule)
fi

EXTRA_RESUME_FLAGS=()
if [[ -n "$START_CHECKPOINT" ]]; then
    EXTRA_RESUME_FLAGS+=(--start_checkpoint "$START_CHECKPOINT")
fi

"$PYTHON_BIN" "$ROOT_DIR/train.py" \
    --source_path "$SOURCE_PATH" \
    --model_path "$MODEL_PATH" \
    --bind_to_mesh \
    --white_background \
    --eval \
    --coord bary \
    --scale_res 0.25 \
    --use_fastgs \
    --use_shortersplatting \
    --iterations 30000 \
    --position_lr_init 0.001 \
    --position_lr_final 0.0001 \
    --position_lr_delay_mult 0.001 \
    --position_lr_max_steps 600000 \
    --feature_lr 0.01 \
    --opacity_lr 0.3 \
    --scaling_lr 0.01 \
    --rotation_lr 0.01 \
    --densification_interval "$DENSIFICATION_INTERVAL" \
    --opacity_reset_interval "$OPACITY_RESET_INTERVAL" \
    --densify_from_iter "$DENSIFY_FROM_ITER" \
    --densify_until_iter "$DENSIFY_UNTIL_ITER" \
    --densify_grad_threshold 0.0005 \
    --flame_expr_lr 0.001 \
    --flame_trans_lr 1e-06 \
    --flame_pose_lr 1e-05 \
    --percent_dense 0.01 \
    --lambda_dssim 0.2 \
    --lambda_entropy "$LAMBDA_ENTROPY" \
    --scale_reset_factor "$SCALE_RESET_FACTOR" \
    --shorter_scale_prune_quantile "$SHORTER_SCALE_PRUNE_QUANTILE" \
    --shorter_scale_prune_factor "$SHORTER_SCALE_PRUNE_FACTOR" \
    --shorter_prune_max_fraction "$SHORTER_PRUNE_MAX_FRACTION" \
    "${EXTRA_SHORTER_FLAGS[@]}" \
    "${EXTRA_RESUME_FLAGS[@]}" \
    --lambda_xyz 0.0 \
    --threshold_xyz 0.5 \
    --lambda_scale 1.0 \
    --threshold_scale 0.6 \
    --lambda_dynamic_offset 0.0 \
    --lambda_laplacian 0.0 \
    --lambda_dynamic_offset_std 0 \
    --texture_start_iter 0 \
    --texture_lr 0.0025 \
    --texture_lambda 0.1 \
    --initial_pc_size 0.04 \
    --initial_pc_number 150 \
    --initial_pc_number_eye 50 \
    --normal_position_lr 5e-06 \
    --lambda_filter 1 \
    --max_scaling 0.5 \
    --fastgs_filter_interval "$FASTGS_FILTER_INTERVAL" \
    --no-bcull \
    --no-depth
