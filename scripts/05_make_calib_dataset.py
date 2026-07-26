#!/usr/bin/env python3
"""
校准集生成脚本

从原始图片目录生成 640x640 letterbox 后的校准图片和 dataset.txt。
主要用于 INT8 可选量化，FP16 不强制需要。

letterbox 参数：
    - 目标尺寸 640x640
    - padding 颜色 114 (灰色)

示例：
    python scripts/05_make_calib_dataset.py \\
        --input-dir /path/to/raw_calib_images \\
        --output-dir outputs/calib/images_640 \\
        --dataset-txt outputs/calib/dataset.txt \\
        --imgsz 640
"""

from __future__ import annotations

import argparse
import os
import sys

# 将 tools/ 加入 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="校准集生成（letterbox 640x640）")
    p.add_argument("--input-dir", required=True, help="原始校准图片目录")
    p.add_argument("--output-dir", required=True, help="校准图片输出目录")
    p.add_argument("--dataset-txt", required=True, help="dataset.txt 输出路径")
    p.add_argument("--imgsz", type=int, default=640, help="目标尺寸 (默认 640)")
    p.add_argument("--max-images", type=int, default=200, help="最大图片数 (默认 200)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import cv2
    except ImportError:
        print("[ERROR] opencv-python 未安装。pip install opencv-python")
        sys.exit(1)

    # 检查输入目录
    if not os.path.isdir(args.input_dir):
        print(f"[ERROR] 输入目录不存在: {args.input_dir}")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 搜集图片
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = []
    for f in sorted(os.listdir(args.input_dir)):
        if os.path.splitext(f)[1].lower() in exts:
            image_files.append(os.path.join(args.input_dir, f))

    if not image_files:
        print(f"[ERROR] 输入目录中没有图片: {args.input_dir}")
        sys.exit(1)

    # 限制数量
    image_files = image_files[: args.max_images]
    print(f"找到 {len(image_files)} 张图片，开始 letterbox 处理...")

    dataset_lines = []

    for i, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [WARN] 无法读取: {img_path}")
            continue

        # letterbox: resize + pad to 640x640, padding=114
        h, w = img.shape[:2]
        scale = min(args.imgsz / w, args.imgsz / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 创建 canvas
        canvas = cv2.copyMakeBorder(
            resized,
            0, args.imgsz - new_h,
            0, args.imgsz - new_w,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        # 保存
        out_name = f"calib_{i:04d}.jpg"
        out_path = os.path.join(args.output_dir, out_name)
        cv2.imwrite(out_path, canvas)

        # 写入 dataset.txt (绝对路径)
        dataset_lines.append(os.path.abspath(out_path) + "\n")

        if (i + 1) % 50 == 0:
            print(f"  已处理 {i + 1}/{len(image_files)}...")

    # 写入 dataset.txt
    dataset_txt_dir = os.path.dirname(args.dataset_txt)
    if dataset_txt_dir:
        os.makedirs(dataset_txt_dir, exist_ok=True)

    with open(args.dataset_txt, "w") as f:
        f.writelines(dataset_lines)

    print(f"\n✓ 完成：{len(dataset_lines)} 张校准图片")
    print(f"  图片目录:  {args.output_dir}")
    print(f"  dataset:   {args.dataset_txt}")


if __name__ == "__main__":
    main()
