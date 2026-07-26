#!/usr/bin/env python3
"""
环境检查脚本

检查训练和导出所需的环境是否就绪。
"""

from __future__ import annotations

import sys


def check_python() -> str:
    v = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"[PYTHON] {v}")
    return v


def check_torch() -> str | None:
    try:
        import torch
        v = torch.__version__
        print(f"[TORCH]  {v}  (CUDA: {torch.cuda.is_available()})")
        return v
    except ImportError:
        print("[TORCH]  NOT INSTALLED")
        return None


def check_ultralytics() -> str | None:
    try:
        import ultralytics
        v = ultralytics.__version__
        print(f"[ULTRA]  {v}")
        return v
    except ImportError:
        print("[ULTRA]  NOT INSTALLED")
        return None


def check_onnx() -> str | None:
    try:
        import onnx
        v = onnx.__version__
        print(f"[ONNX]   {v}")
        return v
    except ImportError:
        print("[ONNX]   NOT INSTALLED (pip install onnx)")
        return None


def check_onnxruntime() -> str | None:
    try:
        import onnxruntime
        v = onnxruntime.__version__
        print(f"[ONNXRT] {v}")
        return v
    except ImportError:
        print("[ONNXRT] NOT INSTALLED (pip install onnxruntime)")
        return None


def check_opencv() -> str | None:
    try:
        import cv2
        v = cv2.__version__
        print(f"[CV2]    {v}")
        return v
    except ImportError:
        print("[CV2]    NOT INSTALLED (pip install opencv-python)")
        return None


def check_rknn() -> str | None:
    try:
        from rknn.api import RKNN
        print("[RKNN]   rknn-toolkit2 available")
        return "available"
    except ImportError:
        print("[RKNN]   NOT INSTALLED — 需要 x86_64 版 rknn-toolkit2 wheel")
        return None


def main() -> None:
    print("=" * 50)
    print("  CUADC RKNN Pipeline — 环境检查")
    print("=" * 50)
    print()

    results = {
        "python": check_python(),
        "torch": check_torch(),
        "ultralytics": check_ultralytics(),
        "onnx": check_onnx(),
        "onnxruntime": check_onnxruntime(),
        "opencv": check_opencv(),
        "rknn": check_rknn(),
    }

    print()
    train_ok = all([results["torch"], results["ultralytics"], results["opencv"]])
    export_ok = all([results["torch"], results["ultralytics"], results["onnx"], results["onnxruntime"]])
    rknn_ok = results["rknn"] is not None

    print("--- 状态 ---")
    print(f"  训练/ONNX导出环境 (conda uav): {'✓ OK' if train_ok else '✗ 缺少依赖'}")
    print(f"  RKNN转换环境 (conda rknn):     {'✓ OK' if rknn_ok else '✗ 未安装 rknn-toolkit2'}")

    if not rknn_ok:
        print()
        print("  如需 RKNN 转换，请创建 rknn 环境：")
        print("    conda create -n rknn python=3.10 -y")
        print("    conda activate rknn")
        print("    pip install rknn_toolkit2-*-cp310-*-linux_x86_64.whl")

    print()
    print("=" * 50)

    sys.exit(0 if (train_ok or rknn_ok) else 1)


if __name__ == "__main__":
    main()
