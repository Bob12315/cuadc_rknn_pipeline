#!/usr/bin/env python3
"""
INT8 RKNN 转换脚本（可选实验）

⚠️ 重要提示：
    INT8 是可选实验流程。
    当前项目默认使用 FP16。
    之前 INT8 出现过 class score 全 0 的问题（scores min/max/mean = 0/0/0, detections=0），
    因此不要把 INT8 作为默认部署模型。

使用前提：
    1. 已安装 rknn-toolkit2 (x86_64 Linux)
    2. 已通过 05_make_calib_dataset.py 生成校准图片和 dataset.txt

示例：
    conda activate rknn

    python scripts/04_convert_rknn_int8_optional.py \\
        --onnx outputs/onnx/cuadc.onnx \\
        --dataset outputs/calib/dataset.txt \\
        --output outputs/rknn/cuadc-int8.rknn
"""

from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ONNX → INT8 RKNN 转换（可选实验）")
    p.add_argument("--onnx", required=True, help="输入 ONNX 路径")
    p.add_argument("--dataset", required=True, help="校准集 dataset.txt 路径")
    p.add_argument("--output", required=True, help="输出 RKNN 路径")
    p.add_argument("--target", default="rk3588", help="目标平台 (默认 rk3588)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  ⚠️  INT8 RKNN 转换（可选实验）")
    print("=" * 60)
    print()
    print("  注意：项目默认使用 FP16。")
    print("  INT8 之前出现过 class score 全 0 的问题。")
    print("  此脚本仅供实验对比，不作为默认部署流程。")
    print("=" * 60)
    print()

    # 检查 rknn-toolkit2
    try:
        from rknn.api import RKNN
    except ImportError:
        print("[ERROR] rknn-toolkit2 未安装。")
        print("  请: conda create -n rknn python=3.10 -y && conda activate rknn")
        print("      pip install rknn_toolkit2-*-cp310-*-linux_x86_64.whl")
        sys.exit(1)

    # 检查输入
    if not os.path.isfile(args.onnx):
        print(f"[ERROR] ONNX 文件不存在: {args.onnx}")
        sys.exit(1)

    if not os.path.isfile(args.dataset):
        print(f"[ERROR] dataset.txt 不存在: {args.dataset}")
        print("  请先用 05_make_calib_dataset.py 生成校准集。")
        sys.exit(1)

    print(f"  onnx:             {args.onnx}")
    print(f"  dataset:          {args.dataset}")
    print(f"  output:            {args.output}")
    print(f"  target_platform:   {args.target}")
    print()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rknn = RKNN()

    ret = rknn.config(
        target_platform=args.target,
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
    )
    if ret != 0:
        print(f"[ERROR] rknn.config 失败, ret={ret}")
        rknn.release()
        sys.exit(1)
    print("✓ rknn.config 完成")

    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        print(f"[ERROR] rknn.load_onnx 失败, ret={ret}")
        rknn.release()
        sys.exit(1)
    print("✓ rknn.load_onnx 完成")

    # 构建 INT8 量化
    ret = rknn.build(do_quantization=True, dataset=args.dataset)
    if ret != 0:
        print(f"[ERROR] rknn.build (INT8) 失败, ret={ret}")
        rknn.release()
        sys.exit(1)
    print("✓ rknn.build 完成 (INT8)")

    ret = rknn.export_rknn(args.output)
    if ret != 0:
        print(f"[ERROR] rknn.export_rknn 失败, ret={ret}")
        rknn.release()
        sys.exit(1)
    print(f"✓ rknn.export_rknn 完成")

    rknn.release()

    if os.path.isfile(args.output):
        size = os.path.getsize(args.output)
        print(f"\n✓ INT8 RKNN 模型已生成: {args.output} ({size:,} bytes)")
        print("  ⚠️ 请在板端充分验证后再考虑部署。")
    else:
        print(f"\n[ERROR] 输出文件未生成: {args.output}")
        sys.exit(1)


if __name__ == "__main__":
    main()
