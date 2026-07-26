#!/usr/bin/env python3
"""
YOLO 训练脚本

使用 Ultralytics YOLO 训练 detect 模型。
默认模型：yolo11n.pt，输入尺寸 640，epochs 150。

示例：
    conda activate uav

    python scripts/01_train_yolo.py \\
        --data configs/cuadc_data_template.yaml \\
        --model yolo11n.pt \\
        --imgsz 640 \\
        --epochs 150 \\
        --batch 16 \\
        --device 0 \\
        --project outputs/train \\
        --name cuadc_yolo11n_640
"""

from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO Detect 训练")
    p.add_argument("--data", required=True, help="数据集 YAML 路径")
    p.add_argument("--model", default="yolo11n.pt", help="预训练模型 (默认 yolo11n.pt)")
    p.add_argument("--imgsz", type=int, default=640, help="输入尺寸 (默认 640)")
    p.add_argument("--epochs", type=int, default=150, help="训练轮数 (默认 150)")
    p.add_argument("--batch", type=int, default=16, help="批次大小 (默认 16)")
    p.add_argument("--device", default="0", help="设备 (默认 0, CPU 用 'cpu')")
    p.add_argument("--project", default="outputs/train", help="输出项目目录")
    p.add_argument("--name", default="cuadc_yolo11n_640", help="训练名称")
    p.add_argument("--patience", type=int, default=50, help="早停 patience")
    p.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    p.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 检查 Ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics 未安装。请先 conda activate uav")
        sys.exit(1)

    # 检查数据集配置文件
    if not os.path.isfile(args.data):
        print(f"[ERROR] 数据集配置文件不存在: {args.data}")
        sys.exit(1)

    print("=" * 60)
    print("  YOLO 训练")
    print("=" * 60)
    print(f"  data:    {args.data}")
    print(f"  model:   {args.model}")
    print(f"  imgsz:   {args.imgsz}")
    print(f"  epochs:  {args.epochs}")
    print(f"  batch:   {args.batch}")
    print(f"  device:  {args.device}")
    print(f"  project: {args.project}")
    print(f"  name:    {args.name}")
    print("=" * 60)
    print()

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        lr0=args.lr0,
        workers=args.workers,
        exist_ok=True,
    )

    # 输出最终模型路径
    best_pt = os.path.join(args.project, args.name, "weights", "best.pt")
    if os.path.isfile(best_pt):
        print(f"\n✓ 训练完成，best.pt: {best_pt}")
    else:
        print(f"\n⚠ best.pt 未在预期路径找到: {best_pt}")
        print("  请检查训练输出日志。")


if __name__ == "__main__":
    main()
