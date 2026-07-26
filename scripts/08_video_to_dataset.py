#!/usr/bin/env python3
"""
从 MP4 视频自动构建 YOLO 检测数据集。

完整流程：
  1. 遍历 videos/ 目录中的 .mp4/.mkv/.avi 等视频文件
  2. 每隔 N 帧（可设置）提取一张图片
  3. 用训练好的 best.pt YOLO 模型对每张图做推理，生成 YOLO 格式标签
  4. 将图片 + 标签按 7:3 比例随机分配到 datasets/cuadc2026 的 train/val
  5. 完成后删除 videos/ 目录下的所有视频文件

用法示例：
  # 使用 conda 环境直接运行
  python scripts/08_video_to_dataset.py

  # 每 10 帧取一张
  python scripts/08_video_to_dataset.py --frame-interval 10

  # 完整参数说明
  python scripts/08_video_to_dataset.py --help

依赖：
  conda 环境 "yolo"（已安装 ultralytics + torch + opencv-python）
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

# ── 默认路径（相对于仓库根目录） ────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
VIDEO_DIR = BASE_DIR / "videos"
MODEL_PATH = BASE_DIR / "outputs/train/cuadc2026/weights/best.pt"
DATASET_DIR = BASE_DIR / "datasets/cuadc2026"

# 支持的视频扩展名
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm"}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="从视频自动构建 YOLO 检测数据集（抽帧 → 推理 → 入库）"
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=VIDEO_DIR,
        help=f"视频输入目录，默认 {VIDEO_DIR}",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help=f"YOLO 模型路径，默认 {MODEL_PATH}",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help=f"数据集根目录，默认 {DATASET_DIR}",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=30,
        help="每隔多少帧提取一张图片（默认 30）",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="训练集比例（默认 0.7）",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO 推理置信度阈值（默认 0.25）",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO 推理图像尺寸（默认 640）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认 42）",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="不删除原始视频（调试用）",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG 保存质量 0-100（默认 95）",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════
# 第 1 步：从视频抽帧
# ══════════════════════════════════════════════════════════════════════

def collect_videos(video_dir: Path) -> list[Path]:
    """收集目录下的所有视频文件。"""
    if not video_dir.exists():
        raise FileNotFoundError(f"视频目录不存在: {video_dir}")
    videos = sorted(
        p for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(f"视频目录中没有找到视频文件: {video_dir}")
    return videos


def extract_frames(
    video_path: Path,
    output_dir: Path,
    frame_interval: int,
    jpeg_quality: int,
) -> list[Path]:
    """
    从单个视频中抽取帧，返回保存的图片路径列表。

    命名规则：{视频名}_{序号:05d}.jpg  例如 camera_xxx_00001.jpg
    """
    video_stem = video_path.stem
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved: list[Path] = []
    frame_idx = 0
    seq = 0

    pbar = tqdm(
        total=total_frames if total_frames > 0 else None,
        desc=f"抽帧 {video_path.name}",
        unit="frame",
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % frame_interval == 0:
                seq += 1
                out_name = f"{video_stem}_{seq:05d}.jpg"
                out_path = output_dir / out_name
                cv2.imwrite(
                    str(out_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
                )
                saved.append(out_path)

            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()

    return saved


# ══════════════════════════════════════════════════════════════════════
# 第 2 步：YOLO 推理生成标签
# ══════════════════════════════════════════════════════════════════════

def run_inference(
    model: YOLO,
    image_paths: list[Path],
    conf: float,
    imgsz: int,
) -> dict[str, list[str]]:
    """
    对图片列表跑批量推理，返回 {图片文件名（不含扩展名）: [标签行, ...]}。

    标签行格式（YOLO）：  class_id cx cy w h
    cx, cy, w, h 已归一化到 [0, 1]。
    """
    labels_map: dict[str, list[str]] = {}

    # 逐张推理，确保每张图都有对应的标签（哪怕为空）
    for img_path in tqdm(image_paths, desc="YOLO 推理", unit="img"):
        results = model.predict(
            source=str(img_path),
            conf=conf,
            imgsz=imgsz,
            verbose=False,
            stream=False,
        )
        # results 是 list[Results]，这里只有一张图
        result = results[0]

        lines: list[str] = []
        if result.boxes is not None:
            # boxes.xywhn → (N, 4) 归一化 [cx, cy, w, h]
            boxes_xywhn = result.boxes.xywhn
            classes = result.boxes.cls
            for cls_id, xywhn in zip(classes, boxes_xywhn):
                cls_id_int = int(cls_id.item())
                cx, cy, w, h = xywhn.tolist()
                lines.append(f"{cls_id_int} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # 去掉扩展名作为 key（与图片命名保持一致）
        stem = img_path.stem
        labels_map[stem] = lines

    return labels_map


# ══════════════════════════════════════════════════════════════════════
# 第 3 步：写标签文件 + 移动图片到 train / val
# ══════════════════════════════════════════════════════════════════════

def save_labels_for_image(
    stem: str,
    label_lines: list[str],
    label_dir: Path,
) -> None:
    """将标签行写入 {stem}.txt。"""
    label_dir.mkdir(parents=True, exist_ok=True)
    label_path = label_dir / f"{stem}.txt"
    label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))


def move_image_to_split(
    img_path: Path,
    dst_dir: Path,
) -> None:
    """移动图片到目标目录。"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / img_path.name
    if dst.exists():
        raise FileExistsError(f"目标文件已存在: {dst}")
    shutil.move(str(img_path), str(dst))


def distribute_to_dataset(
    image_paths: list[Path],
    labels_map: dict[str, list[str]],
    dataset_dir: Path,
    train_ratio: float,
    seed: int,
) -> tuple[int, int]:
    """
    将图片 + 标签按比例分配到 train / val。

    返回 (train_count, val_count)。
    """
    images_train = dataset_dir / "images" / "train"
    images_val = dataset_dir / "images" / "val"
    labels_train = dataset_dir / "labels" / "train"
    labels_val = dataset_dir / "labels" / "val"

    # 打乱
    rng = random.Random(seed)
    shuffled = image_paths[:]
    rng.shuffle(shuffled)

    split_idx = round(len(shuffled) * train_ratio)
    train_paths = shuffled[:split_idx]
    val_paths = shuffled[split_idx:]

    # 处理 train
    for img in tqdm(train_paths, desc="写入 train", unit="img"):
        stem = img.stem
        lines = labels_map.get(stem, [])
        save_labels_for_image(stem, lines, labels_train)
        move_image_to_split(img, images_train)

    # 处理 val
    for img in tqdm(val_paths, desc="写入 val", unit="img"):
        stem = img.stem
        lines = labels_map.get(stem, [])
        save_labels_for_image(stem, lines, labels_val)
        move_image_to_split(img, images_val)

    return len(train_paths), len(val_paths)


# ══════════════════════════════════════════════════════════════════════
# 第 4 步：清理
# ══════════════════════════════════════════════════════════════════════

def cleanup_videos(video_dir: Path) -> None:
    """删除视频目录下的所有视频文件（不删除目录本身）。"""
    for p in video_dir.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            p.unlink()
            print(f"  已删除: {p.name}")


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── 基础校验 ──
    if not args.model.exists():
        raise FileNotFoundError(f"模型文件不存在: {args.model}")
    if args.frame_interval <= 0:
        raise ValueError("--frame-interval 必须 > 0")
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio 必须在 (0, 1) 范围内")
    if not 0 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality 必须在 0-100")

    # ── 加载模型 ──
    print(f"加载模型: {args.model}")
    model = YOLO(str(args.model))
    # 从模型中获取类别数，做个基础校验
    nc = getattr(model.model, "nc", None)
    if nc is not None and hasattr(model, "names"):
        print(f"  模型类别数: {nc}")
        print(f"  类别名称: {model.names}")

    # ── 1. 收集视频 ──
    videos = collect_videos(args.video_dir)
    print(f"\n找到 {len(videos)} 个视频文件:")
    for v in videos:
        print(f"  - {v.name}")

    # ── 临时目录用于存放抽取的图片（稍后会移动到数据集） ──
    tmp_dir = args.video_dir / "_extracted_frames"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    # ── 2. 抽帧 ──
    all_frames: list[Path] = []
    for v in videos:
        frames = extract_frames(
            video_path=v,
            output_dir=tmp_dir,
            frame_interval=args.frame_interval,
            jpeg_quality=args.jpeg_quality,
        )
        all_frames.extend(frames)
        print(f"  {v.name}: 抽取 {len(frames)} 张")

    print(f"\n共抽取 {len(all_frames)} 张图片")

    if not all_frames:
        print("没有抽到任何图片，退出。")
        return

    # ── 3. YOLO 推理 ──
    print(f"\n开始 YOLO 推理（conf={args.conf}, imgsz={args.imgsz}）...")
    labels_map = run_inference(
        model=model,
        image_paths=all_frames,
        conf=args.conf,
        imgsz=args.imgsz,
    )

    # 统计有标注 / 无标注的图片数
    annotated = sum(1 for lines in labels_map.values() if lines)
    empty = len(labels_map) - annotated
    print(f"  有检测框: {annotated} 张, 无检测框: {empty} 张")

    # ── 4. 分配到数据集 ──
    print(f"\n按 {args.train_ratio}:{1 - args.train_ratio} 分配到 train/val ...")
    train_cnt, val_cnt = distribute_to_dataset(
        image_paths=all_frames,
        labels_map=labels_map,
        dataset_dir=args.dataset_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    # 删除临时目录
    shutil.rmtree(tmp_dir)

    print(f"\n{'='*50}")
    print(f"数据集入库完成！")
    print(f"  train: {train_cnt} 张  →  {args.dataset_dir / 'images' / 'train'}")
    print(f"  val:   {val_cnt} 张  →  {args.dataset_dir / 'images' / 'val'}")
    print(f"  train 标签: {args.dataset_dir / 'labels' / 'train'}")
    print(f"  val 标签:   {args.dataset_dir / 'labels' / 'val'}")
    print(f"{'='*50}")

    # ── 5. 清理视频 ──
    if not args.no_cleanup:
        print("\n清理 videos 目录...")
        cleanup_videos(args.video_dir)
        print("清理完成。")
    else:
        print("\n--no-cleanup 已设置，保留原始视频。")


if __name__ == "__main__":
    main()
