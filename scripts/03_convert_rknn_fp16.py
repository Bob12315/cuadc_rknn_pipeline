#!/usr/bin/env python3
"""
FP16 RKNN 转换脚本（主流程）

使用 rknn-toolkit2 将 ONNX 转换为 FP16 RKNN 模型。

重要：
  - 不做 INT8 量化 (do_quantization=False)
  - target_platform 必须是 rk3588
  - 在 x86_64 Linux 上运行，不要在 RK3588 ARM 板上运行

示例：
    conda activate rknn

    python scripts/03_convert_rknn_fp16.py \\
        --onnx outputs/onnx/cuadc.onnx \\
        --output outputs/rknn/cuadc-fp16.rknn
"""

from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ONNX → FP16 RKNN 转换")
    p.add_argument("--onnx", required=True, help="输入 ONNX 路径")
    p.add_argument("--output", required=True, help="输出 RKNN 路径")
    p.add_argument("--target", default="rk3588", help="目标平台 (默认 rk3588)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 检查 rknn-toolkit2
    try:
        from rknn.api import RKNN
    except ImportError:
        print("=" * 60)
        print("  [ERROR] rknn-toolkit2 未安装")
        print("=" * 60)
        print()
        print("  RKNN 转换需要 x86_64 Linux 版本的 rknn-toolkit2。")
        print()
        print("  请执行以下步骤创建 rknn 环境：")
        print()
        print("    conda create -n rknn python=3.10 -y")
        print("    conda activate rknn")
        print("    pip install rknn_toolkit2-*-cp310-*-linux_x86_64.whl")
        print()
        print("  注意：不要在 RK3588 ARM 板上运行此脚本。")
        print("=" * 60)
        sys.exit(1)

    # 检查 ONNX 文件
    if not os.path.isfile(args.onnx):
        print(f"[ERROR] ONNX 文件不存在: {args.onnx}")
        sys.exit(1)

    print("=" * 60)
    print("  FP16 RKNN 转换")
    print("=" * 60)
    print(f"  onnx:             {args.onnx}")
    print(f"  output:            {args.output}")
    print(f"  target_platform:   {args.target}")
    print(f"  do_quantization:   False (FP16)")
    print("=" * 60)
    print()

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 创建 RKNN 对象
    rknn = RKNN()

    # 配置
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

    # 加载 ONNX
    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        print(f"[ERROR] rknn.load_onnx 失败, ret={ret}")
        rknn.release()
        sys.exit(1)
    print("✓ rknn.load_onnx 完成")

    # 构建 RKNN (FP16, 不量化)
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f"[ERROR] rknn.build 失败, ret={ret}")
        rknn.release()
        sys.exit(1)
    print("✓ rknn.build 完成 (FP16)")

    # 导出 RKNN
    ret = rknn.export_rknn(args.output)
    if ret != 0:
        print(f"[ERROR] rknn.export_rknn 失败, ret={ret}")
        rknn.release()
        sys.exit(1)
    print(f"✓ rknn.export_rknn 完成")

    rknn.release()

    # 确认输出
    if os.path.isfile(args.output):
        size = os.path.getsize(args.output)
        print(f"\n✓ FP16 RKNN 模型已生成: {args.output} ({size:,} bytes)")
    else:
        print(f"\n[ERROR] 输出文件未生成: {args.output}")
        sys.exit(1)


if __name__ == "__main__":
    main()
