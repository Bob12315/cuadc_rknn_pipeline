#!/usr/bin/env python3
"""
公共工具函数

提供 letterbox、ensure_dir、copy_if_needed、print_header、resolve_path 等工具函数。

letterbox 说明：
    - OpenCV BGR 输入
    - resize/pad 到 640x640
    - padding=114
    - 输出 RGB uint8, shape=(1, 640, 640, 3)
"""

from __future__ import annotations

import os
import shutil
from typing import Tuple

import numpy as np


def letterbox(
    img: np.ndarray,
    target_size: int = 640,
    padding: int = 114,
    return_rgb: bool = True,
) -> np.ndarray:
    """
    letterbox 预处理：将任意尺寸图片 resize + pad 到 target_size × target_size。

    Args:
        img: OpenCV BGR uint8 图片 (H, W, 3)
        target_size: 目标正方形边长，默认 640
        padding: padding 颜色值，默认 114
        return_rgb: 输出 RGB (默认) 否则 BGR

    Returns:
        处理后的图片，shape (1, target_size, target_size, 3), uint8
    """
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

    if return_rgb:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    return np.expand_dims(canvas, axis=0).astype(np.uint8)


def ensure_dir(path: str) -> str:
    """确保目录存在，如果不存在则创建。返回目录路径。"""
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path


def copy_if_needed(src: str, dst: str, overwrite: bool = False) -> bool:
    """
    如果目标文件不存在（或 overwrite=True），从 src 复制到 dst。
    返回 True 表示已复制。
    """
    if not os.path.isfile(src):
        raise FileNotFoundError(f"源文件不存在: {src}")

    dst_dir = os.path.dirname(dst)
    if dst_dir:
        ensure_dir(dst_dir)

    if os.path.isfile(dst) and not overwrite:
        return False

    shutil.copy2(src, dst)
    return True


def print_header(title: str, width: int = 60) -> None:
    """打印格式化的标题。"""
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print()


def resolve_path(base: str, *parts: str) -> str:
    """基于 base 解析相对路径，返回绝对路径。"""
    return os.path.abspath(os.path.join(base, *parts))
