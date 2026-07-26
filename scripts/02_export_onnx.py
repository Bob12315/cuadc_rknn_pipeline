#!/usr/bin/env python3
"""
ONNX 导出脚本

从 .pt 导出固定 640x640 ONNX。

参数：
    format=onnx, imgsz=640, opset=12, simplify=True, dynamic=False

示例：
    conda activate uav

    python scripts/02_export_onnx.py \\
        --weights outputs/train/cuadc_yolo11n_640/weights/best.pt \\
        --imgsz 640 \\
        --opset 12 \\
        --output outputs/onnx/cuadc.onnx
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO → ONNX 导出")
    p.add_argument("--weights", required=True, help="输入 .pt 路径")
    p.add_argument("--imgsz", type=int, default=640, help="输入尺寸 (默认 640)")
    p.add_argument("--opset", type=int, default=12, help="ONNX opset (默认 12)")
    p.add_argument("--output", required=True, help="输出 ONNX 路径")
    p.add_argument("--dynamic", action="store_true", default=False, help="动态 batch (默认 False)")
    p.add_argument("--no-simplify", action="store_true", default=False, help="不简化模型")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 检查 Ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics 未安装。请先 conda activate uav")
        sys.exit(1)

    # 检查权重文件
    if not os.path.isfile(args.weights):
        print(f"[ERROR] 权重文件不存在: {args.weights}")
        sys.exit(1)

    print("=" * 60)
    print("  ONNX 导出")
    print("=" * 60)
    print(f"  weights:    {args.weights}")
    print(f"  imgsz:      {args.imgsz}")
    print(f"  opset:      {args.opset}")
    print(f"  simplify:   {not args.no_simplify}")
    print(f"  dynamic:    {args.dynamic}")
    print(f"  output:     {args.output}")
    print("=" * 60)
    print()

    model = YOLO(args.weights)

    # Ultralytics export 默认输出到权重目录，命名为 <stem>.onnx
    export_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
        dynamic=args.dynamic,
    )
    print(f"\nUltralytics 导出到: {export_path}")

    # 复制/移动到目标路径
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if os.path.abspath(export_path) != os.path.abspath(args.output):
        shutil.copy2(export_path, args.output)
        print(f"已复制到: {args.output}")

    print(f"\n✓ ONNX 导出完成: {args.output}")


if __name__ == "__main__":
    main()
