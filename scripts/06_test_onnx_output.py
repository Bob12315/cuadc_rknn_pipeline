#!/usr/bin/env python3
"""
ONNX 输出检测脚本

使用 onnxruntime 加载 ONNX 模型，输入一张图片做 640x640 letterbox，
打印输出 shape 并检查是否符合预期。

对 nc=3 的 detect 模型，期望输出 shape = (1, 7, 8400)，即 (1, 4+nc, 8400)。

示例：
    conda activate uav

    python scripts/06_test_onnx_output.py \\
        --onnx outputs/onnx/cuadc.onnx \\
        --image /path/to/test_image.jpg
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ONNX 输出检测")
    p.add_argument("--onnx", required=True, help="ONNX 模型路径")
    p.add_argument("--image", required=True, help="测试图片路径")
    p.add_argument("--imgsz", type=int, default=640, help="输入尺寸 (默认 640)")
    p.add_argument("--nc", type=int, default=3, help="类别数 (默认 3)")
    return p.parse_args()


def letterbox(img: np.ndarray, target_size: int = 640, padding: int = 114) -> np.ndarray:
    """OpenCV BGR 输入 → resize/pad 到 target_size × target_size → RGB uint8 (1, H, W, 3)"""
    import cv2

    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = cv2.copyMakeBorder(
        resized,
        0, target_size - new_h,
        0, target_size - new_w,
        cv2.BORDER_CONSTANT,
        value=(padding, padding, padding),
    )
    # BGR → RGB
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    # (H, W, 3) → (1, H, W, 3) uint8
    return np.expand_dims(rgb, axis=0).astype(np.uint8)


def main() -> None:
    args = parse_args()

    # 检查依赖
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ERROR] onnxruntime 未安装。pip install onnxruntime")
        sys.exit(1)

    try:
        import cv2
    except ImportError:
        print("[ERROR] opencv-python 未安装。pip install opencv-python")
        sys.exit(1)

    # 检查输入
    if not os.path.isfile(args.onnx):
        print(f"[ERROR] ONNX 文件不存在: {args.onnx}")
        sys.exit(1)

    if not os.path.isfile(args.image):
        print(f"[ERROR] 图片不存在: {args.image}")
        sys.exit(1)

    print("=" * 60)
    print("  ONNX 输出检测")
    print("=" * 60)
    print(f"  onnx:  {args.onnx}")
    print(f"  image: {args.image}")
    print(f"  imgsz: {args.imgsz}")
    print(f"  nc:    {args.nc}")
    print("=" * 60)
    print()

    # 加载 ONNX
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"模型输入: {input_name} {input_shape}")

    output_name = session.get_outputs()[0].name
    output_shape = session.get_outputs()[0].shape
    print(f"模型输出: {output_name} {output_shape}")
    print()

    # 读取图片并 letterbox
    img = cv2.imread(args.image)
    if img is None:
        print(f"[ERROR] 无法读取图片: {args.image}")
        sys.exit(1)

    input_data = letterbox(img, target_size=args.imgsz)
    # 转换为 float32 并归一化到 [0, 1]
    input_data_fp32 = input_data.astype(np.float32) / 255.0
    print(f"处理后输入 shape: {input_data_fp32.shape}")

    # 推理
    outputs = session.run([output_name], {input_name: input_data_fp32})
    out = outputs[0]
    print(f"\n实际输出 shape: {out.shape}")

    # 检查
    expected_channels = 4 + args.nc
    expected_shape = (1, expected_channels, 8400)

    if out.shape == expected_shape:
        print(f"\n✓ 输出 shape 符合预期: {expected_shape}")
    else:
        print(f"\n⚠ WARNING: 输出 shape 不符合预期！")
        print(f"  实际: {out.shape}")
        print(f"  期望: {expected_shape}")
        print(f"  请检查：")
        print(f"    - 是否是 detect 模型（不是 segment/pose/classify）")
        print(f"    - dynamic 是否设为 False")
        print(f"    - imgsz 是否为 640")
        print(f"    - 类别数 nc 是否正确（当前假设 nc={args.nc}）")

    # 打印一些统计信息
    print(f"\n输出统计:")
    print(f"  min:  {out.min():.6f}")
    print(f"  max:  {out.max():.6f}")
    print(f"  mean: {out.mean():.6f}")


if __name__ == "__main__":
    main()
