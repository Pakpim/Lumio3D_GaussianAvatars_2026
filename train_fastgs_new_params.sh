#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/rpool/data/icy/venv/bin/python}"
SOURCE_PATH="${SOURCE_PATH:-/rpool/data/icy/dataset/bird_v01}"
RUN_NAME="${RUN_NAME:-fastgs_new_params_30000}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/output/$RUN_NAME}"
FASTGS_FILTER_INTERVAL="${FASTGS_FILTER_INTERVAL:-4}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] Python executable not found: $PYTHON_BIN"
    exit 1
fi

mkdir -p "$MODEL_PATH"

echo "[INFO] Starting FastGS training with new params"
echo "[INFO] source_path: $SOURCE_PATH"
echo "[INFO] model_path:  $MODEL_PATH"
echo "[INFO] fastgs_filter_interval: $FASTGS_FILTER_INTERVAL"

# Note: lambda_opacity is not a supported CLI option in lumio_base/train.py.
"$PYTHON_BIN" "$ROOT_DIR/train.py" \
    --source_path "$SOURCE_PATH" \
    --model_path "$MODEL_PATH" \
    --bind_to_mesh \
    --white_background \
    --eval \
    --coord bary \
    --scale_res 0.25 \
    --use_fastgs \
    --iterations 30000 \
    --position_lr_init 0.001 \
    --position_lr_final 0.0001 \
    --position_lr_delay_mult 0.001 \
    --position_lr_max_steps 600000 \
    --feature_lr 0.01 \
    --opacity_lr 0.3 \
    --scaling_lr 0.01 \
    --rotation_lr 0.01 \
    --densification_interval 30000 \
    --opacity_reset_interval 30000 \
    --densify_from_iter 5000 \
    --densify_until_iter 20000 \
    --densify_grad_threshold 0.0005 \
    --flame_expr_lr 0.001 \
    --flame_trans_lr 1e-06 \
    --flame_pose_lr 1e-05 \
    --percent_dense 0.01 \
    --lambda_dssim 0.2 \
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
