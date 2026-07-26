#!/bin/bash
# ============================================================
#  训练 datasets_dangerous → ONNX → FP16 RKNN
#  使用方法: bash train_dangerous.sh
# ============================================================

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ENV="${TRAIN_ENV:-uav}"
RKNN_ENV="${RKNN_ENV:-rknn}"
cd "$REPO_DIR"

# --- 激活训练环境 ---
eval "$(conda shell.bash hook)"
conda activate "$TRAIN_ENV"

# ============================================================
# 第一步: 训练 YOLO11n
# ============================================================
echo "============================================"
echo "  [1/3] 训练 YOLO11n ..."
echo "============================================"
python scripts/01_train_yolo.py \
  --data datasets/datasets_dangerous/datasets_dangerous.yaml \
  --model yolo11n.pt \
  --imgsz 640 \
  --epochs 150 \
  --batch 16 \
  --device 0 \
  --project outputs/train \
  --name dangerous \
  --workers 4

# ============================================================
# 第二步: 导出 ONNX
# ============================================================
echo ""
echo "============================================"
echo "  [2/3] 导出 ONNX ..."
echo "============================================"
python scripts/02_export_onnx.py \
  --weights outputs/train/dangerous/weights/best.pt \
  --imgsz 640 \
  --opset 12 \
  --output outputs/onnx/dangerous.onnx

# ============================================================
# 第三步: 转换 FP16 RKNN
# ============================================================
echo ""
echo "============================================"
echo "  [3/3] 转换 FP16 RKNN ..."
echo "============================================"
conda run -n "$RKNN_ENV" python scripts/03_convert_rknn_fp16.py \
  --onnx outputs/onnx/dangerous.onnx \
  --output outputs/rknn/dangerous-fp16.rknn

# ============================================================
# 完成
# ============================================================
echo ""
echo "============================================"
echo "  ✓ 全部完成!"
echo "============================================"
echo "  best.pt:   outputs/train/dangerous/weights/best.pt"
echo "  onnx:      outputs/onnx/dangerous.onnx"
echo "  rknn:      outputs/rknn/dangerous-fp16.rknn"
