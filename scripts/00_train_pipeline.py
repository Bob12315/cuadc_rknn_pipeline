#!/usr/bin/env python3
"""
通用训练流水线：训练 → ONNX → RKNN，一条命令完成。

默认训练 cuadc2026 数据集，也可指定其他数据集和模型：
  python scripts/00_train_pipeline.py
  python scripts/00_train_pipeline.py --data datasets/other/other.yaml
  python scripts/00_train_pipeline.py --data datasets/other/other.yaml --model some.pt
"""

from __future__ import annotations

import os
import argparse
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # cuadc_rknn_pipeline/
SCRIPTS = BASE_DIR / "scripts"
OUTPUTS = BASE_DIR / "outputs"

# 默认路径
DEFAULT_DATA = str(BASE_DIR / "datasets" / "cuadc2026" / "cuadc2026.yaml")
DEFAULT_MODEL = str(BASE_DIR / "yolo11n.pt")

# 禁止 ~/.local 用户 site-packages 干扰 conda 环境
CLEAN_ENV = {**os.environ, "PYTHONNOUSERSITE": "1"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="通用训练流水线：训练 → ONNX → RKNN")
    p.add_argument("--data", default=DEFAULT_DATA, help=f"数据集 YAML，默认 {DEFAULT_DATA}")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"预训练模型，默认 {DEFAULT_MODEL}")
    p.add_argument("--name", default=None, help="项目名，默认从 --data 文件名推导")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--train-env", default="uav", help="训练/导出使用的 Conda 环境")
    p.add_argument("--rknn-env", default="rknn", help="RKNN 转换使用的 Conda 环境")
    return p.parse_args()


def run(cmd: list[str], step: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  [{step}] {' '.join(cmd)}")
    print(f"{'=' * 60}\n")
    subprocess.run(cmd, check=True, env=CLEAN_ENV)


def main() -> None:
    args = parse_args()
    name = args.name or Path(args.data).stem

    train_out = OUTPUTS / "train" / name
    onnx_out = OUTPUTS / "onnx" / f"{name}.onnx"
    rknn_out = OUTPUTS / "rknn" / f"{name}-fp16.rknn"
    conda_train = ["conda", "run", "--no-capture-output", "-n", args.train_env]
    conda_rknn = ["conda", "run", "--no-capture-output", "-n", args.rknn_env]

    print(f"data:   {args.data}")
    print(f"model:  {args.model}")
    print(f"name:   {name}")
    print(f"输出:   {train_out / 'weights' / 'best.pt'}")
    print(f"       → {onnx_out}")
    print(f"       → {rknn_out}")

    # Step 1: 训练
    run(conda_train + [
        "python", str(SCRIPTS / "01_train_yolo.py"),
        "--data", args.data,
        "--model", args.model,
        "--imgsz", str(args.imgsz),
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--device", args.device,
        "--project", str(OUTPUTS / "train"),
        "--name", name,
        "--workers", str(args.workers),
    ], "1/3 训练")

    # Step 2: 导出 ONNX
    run(conda_train + [
        "python", str(SCRIPTS / "02_export_onnx.py"),
        "--weights", str(train_out / "weights" / "best.pt"),
        "--imgsz", str(args.imgsz),
        "--opset", "12",
        "--output", str(onnx_out),
    ], "2/3 导出 ONNX")

    # Step 3: 转换 RKNN
    run(conda_rknn + [
        "python", str(SCRIPTS / "03_convert_rknn_fp16.py"),
        "--onnx", str(onnx_out),
        "--output", str(rknn_out),
    ], "3/3 转换 RKNN")

    print(f"\n{'=' * 60}")
    print(f"  ✓ 全部完成!")
    print(f"{'=' * 60}")
    print(f"  best.pt:  {train_out / 'weights' / 'best.pt'}")
    print(f"  onnx:     {onnx_out}")
    print(f"  rknn:     {rknn_out}")


if __name__ == "__main__":
    main()
