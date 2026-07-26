#!/usr/bin/env python3
"""
RKNN 模型信息查看脚本

读取并打印 RKNN 模型的基本信息：文件大小、路径、生成时间等。
如果 Toolkit API 支持，尝试打印模型输入输出信息。

示例：
    conda activate rknn

    python scripts/07_print_rknn_model_info.py \\
        --rknn outputs/rknn/cuadc-fp16.rknn
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RKNN 模型信息查看")
    p.add_argument("--rknn", required=True, help="RKNN 模型路径")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.rknn):
        print(f"[ERROR] RKNN 文件不存在: {args.rknn}")
        sys.exit(1)

    # 文件信息
    stat = os.stat(args.rknn)
    size = stat.st_size
    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))

    print("=" * 60)
    print("  RKNN 模型信息")
    print("=" * 60)
    print(f"  路径:       {os.path.abspath(args.rknn)}")
    print(f"  文件名:     {os.path.basename(args.rknn)}")
    print(f"  大小:       {size:,} bytes ({size / 1024:.1f} KB)")
    print(f"  修改时间:   {mtime}")
    print("=" * 60)
    print()

    # 尝试用 rknn-toolkit2 加载并获取更多信息
    try:
        from rknn.api import RKNN

        rknn = RKNN()
        ret = rknn.load_rknn(path=args.rknn)
        if ret != 0:
            print(f"[WARN] rknn.load_rknn 返回 ret={ret}，API 可能不支持详细信息查询。")
            rknn.release()
            return

        # 尝试获取模型信息
        try:
            # rknn-toolkit2 某些版本支持 query 方法
            info = rknn.query(detail=True)
            print("模型详细信息:")
            if isinstance(info, dict):
                for k, v in info.items():
                    print(f"  {k}: {v}")
            else:
                print(f"  {info}")
        except AttributeError:
            print("[INFO] 当前 Toolkit API 不支持 query() 方法。")
        except Exception as e:
            print(f"[INFO] 无法获取详细模型信息: {e}")

        rknn.release()

    except ImportError:
        print("[INFO] rknn-toolkit2 未安装，仅显示文件基本信息。")
        print("  如需详细信息，请安装 x86_64 版 rknn-toolkit2。")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
