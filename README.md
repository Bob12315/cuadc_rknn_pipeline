# CUADC RKNN Pipeline

> 数据集训练 YOLO Detect 模型，并导出 RK3588 可用的 **FP16 RKNN** 模型。

仓库只保存源码、配置模板和文档。数据集、视频、模型权重、训练输出与日志均由
`.gitignore` 排除，需要在本地自行准备。

## 快速安装

建议使用 Python 3.10，并根据训练机器的 CUDA 版本先安装 PyTorch：

```bash
python -m pip install -r requirements.txt
python scripts/00_check_env.py
```

RKNN 转换环境需另外安装 x86_64 Linux 版本的 `rknn-toolkit2`，详见
[`docs/train_export_convert.md`](docs/train_export_convert.md)。

## 目录结构

```
cuadc_rknn_pipeline/
├── README.md                          # 本文件
├── requirements.txt                  # 通用 Python 依赖（不含 PyTorch/RKNN）
├── configs/
│   ├── cuadc_data_template.yaml       # 数据集配置模板
│   ├── train_yolo11n.yaml             # 训练超参配置
│   └── rknn_config.yaml               # RKNN 转换配置
├── scripts/
│   ├── 00_train_pipeline.py           # 通用训练流水线 (训练→ONNX→RKNN)
│   ├── 00_check_env.py                # 环境检查
│   ├── 01_train_yolo.py               # YOLO 训练
│   ├── 02_export_onnx.py              # ONNX 导出
│   ├── 03_convert_rknn_fp16.py        # FP16 RKNN 转换（主流程）
│   ├── 04_convert_rknn_int8_optional.py  # INT8 RKNN 转换（可选实验）
│   ├── 05_make_calib_dataset.py       # 校准集生成
│   ├── 06_test_onnx_output.py         # ONNX 输出检测
│   ├── 07_print_rknn_model_info.py    # RKNN 模型信息查看
│   └── 08_video_to_dataset.py         # 视频自动标注入库
├── tools/
│   └── common.py                      # 公共工具函数
├── outputs/
│   ├── onnx/                          # ONNX 模型输出
│   ├── rknn/                          # RKNN 模型输出
│   ├── logs/                          # 日志
│   └── train/                         # 训练输出
└── docs/
    ├── dataset_format.md              # 数据集格式说明
    ├── train_export_convert.md        # 训练/导出/转换 完整操作
    ├── rknn_fp16_deploy.md            # RK3588 部署说明
    └── troubleshooting.md             # 故障排查
```

## 重要约定

### 默认使用 FP16 RKNN

- **默认部署模型：FP16**
- INT8 是可选项，**不作为主模型**
- 之前 INT8 出现过 `class score 全 0 / detections=0` 的问题，因此 **不要把 INT8 作为默认部署模型**
- INT8 仅保留为可选实验脚本

### 硬件分工

| 阶段 | 执行平台 |
|------|----------|
| 训练 (train) | x86_64 Linux (GPU) |
| ONNX 导出 (export) | x86_64 Linux |
| RKNN 转换 (convert) | **x86_64 Linux** ← 必须！ |
| RKNN 推理 (inference) | **RK3588 ARM 板** |

> ⚠️ RKNN 转换必须在 x86_64 Linux 上执行，不能在 RK3588 ARM 板上执行。
> RK3588 只运行 `.rknn` 文件，不负责转换。

## 从视频构建数据集

> 将 MP4 视频自动抽帧 → YOLO 推理标注 → 入库到 `datasets/cuadc2026/`

### 流程说明

```
videos/*.mp4
    → 每隔 N 帧抽取一张 JPEG
    → best.pt YOLO 推理生成标签（YOLO txt 格式）
    → 按 7:3 随机分配到 datasets/cuadc2026/images/{train,val}/
    → 标签写入 datasets/cuadc2026/labels/{train,val}/
    → 删除 videos/ 下已处理的视频
```

### 使用方式

```bash
# 激活 yolo 环境（需要 ultralytics + torch + opencv-python）
conda activate yolo

# 默认参数运行：每 30 帧抽一张，7:3 拆分，置信度 0.25
python scripts/08_video_to_dataset.py

# 每 5 帧抽一张，置信度调到 0.8
python scripts/08_video_to_dataset.py --frame-interval 5 --conf 0.8

# 调试模式：不删除原始视频
python scripts/08_video_to_dataset.py --no-cleanup
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--video-dir` | `videos/` | 视频输入目录 |
| `--model` | `outputs/train/cuadc2026/weights/best.pt` | YOLO 模型路径 |
| `--dataset-dir` | `datasets/cuadc2026/` | 数据集根目录 |
| `--frame-interval` | `30` | 每隔多少帧抽取一张图片 |
| `--train-ratio` | `0.7` | 训练集比例 |
| `--conf` | `0.25` | YOLO 推理置信度阈值 |
| `--imgsz` | `640` | 推理图像尺寸 |
| `--seed` | `42` | 随机种子，可复现拆分结果 |
| `--jpeg-quality` | `95` | JPEG 保存质量 (0-100) |
| `--no-cleanup` | `False` | 不删除原始视频（调试用） |

### 输出示例

```
加载模型: outputs/train/cuadc2026/weights/best.pt
  模型类别数: 11
  类别名称: {0: 'baozha', 1: 'shenghua', ...}

找到 1 个视频文件:
  - camera_20260708_200727_225682.mp4

抽帧 camera_20260708_200727_225682.mp4: 100%|████████| 1518/1518
  camera_20260708_200727_225682.mp4: 抽取 51 张

共抽取 51 张图片

开始 YOLO 推理（conf=0.25, imgsz=640）...
YOLO 推理: 100%|████████| 51/51
  有检测框: 48 张, 无检测框: 3 张

按 0.7:0.3 分配到 train/val ...
写入 train: 100%|████████| 36/36
写入 val: 100%|████████| 15/15

==================================================
数据集入库完成！
  train: 36 张  →  datasets/cuadc2026/images/train
  val:   15 张  →  datasets/cuadc2026/images/val
  train 标签: datasets/cuadc2026/labels/train
  val 标签:   datasets/cuadc2026/labels/val
==================================================

清理 videos 目录...
  已删除: camera_20260708_200727_225682.mp4
清理完成。
```

### 注意

- 抽取的图片存放在临时目录，入库完成后自动清理
- **无检测框的图片（空标签）也会入库**，标签文件为空，防止数据浪费
- 入库时检测文件名冲突，避免覆盖已有数据
- 需要 conda 环境 `yolo`（含 ultralytics + torch + opencv-python）

## 推荐主流程

```
dataset.yaml
    → train best.pt
    → export best.onnx
    → convert cuadc-fp16.rknn
    → scp 到 RK3588 的 uav_system-rk3588/data/models/
```

## 一键训练流水线

> `00_train_pipeline.py` — 训练 → ONNX → RKNN，默认训练 cuadc2026 数据集。

```bash
cd /path/to/cuadc_rknn_pipeline

# 默认：训练 cuadc2026 数据集，模型 yolo11n.pt
python scripts/00_train_pipeline.py

# 指定数据集
python scripts/00_train_pipeline.py \
  --data datasets/cuadc2026/cuadc2026.yaml

python scripts/00_train_pipeline.py \
  --data datasets/gazebo_dataset/gazebo_dataset.yaml

# 换数据集 + 换模型
python scripts/00_train_pipeline.py \
  --data datasets/other/other.yaml \
  --model yolo11n.pt
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | `datasets/cuadc2026/cuadc2026.yaml` | 数据集 YAML 路径 |
| `--model` | `yolo11n.pt` | 预训练模型路径 |
| `--name` | 从 `--data` 推导 | 项目名称，决定输出子目录 |
| `--imgsz` | `640` | 输入尺寸 |
| `--epochs` | `150` | 训练轮数 |
| `--batch` | `16` | 批次大小 |
| `--device` | `0` | GPU 编号 |
| `--workers` | `4` | DataLoader 进程数 |
| `--train-env` | `uav` | 训练与 ONNX 导出的 Conda 环境 |
| `--rknn-env` | `rknn` | RKNN 转换的 Conda 环境 |

输出：
```
outputs/train/{name}/weights/best.pt
outputs/onnx/{name}.onnx
outputs/rknn/{name}-fp16.rknn
```

## 快速开始（分步命令链）

> 下面是各阶段手动命令，适合调试。日常使用推荐上面的 `00_train_pipeline.py`。

### 0. 环境检查

```bash
conda activate uav
python scripts/00_check_env.py
```

### 1. 准备数据集配置

编辑 `configs/cuadc_data_template.yaml`，填写你的数据集路径：

```yaml
path: /absolute/path/to/cuadc_dataset
train: images/train
val: images/val
test: images/test
names:
  0: Target
  1: bucket
  2: class_2
```

### 2. 训练

```bash
conda activate uav

python scripts/01_train_yolo.py \
  --data configs/cuadc_data_template.yaml \
  --model yolo11n.pt \
  --imgsz 640 \
  --epochs 150 \
  --batch 16 \
  --device 0 \
  --project outputs/train \
  --name cuadc_yolo11n_640
```

训练输出模型：`outputs/train/cuadc_yolo11n_640/weights/best.pt`

### 3. 导出 ONNX

```bash
conda activate uav

python scripts/02_export_onnx.py \
  --weights outputs/train/cuadc_yolo11n_640/weights/best.pt \
  --imgsz 640 \
  --opset 12 \
  --output outputs/onnx/cuadc.onnx
```

### 4. 检测 ONNX 输出

```bash
conda activate uav

python scripts/06_test_onnx_output.py \
  --onnx outputs/onnx/cuadc.onnx \
  --image /path/to/test_image.jpg
```

期望输出 shape：`(1, 7, 8400)` （nc=3 时）

### 5. 转换 FP16 RKNN

```bash
conda activate rknn

python scripts/03_convert_rknn_fp16.py \
  --onnx outputs/onnx/cuadc.onnx \
  --output outputs/rknn/cuadc-fp16.rknn
```

### 6. 查看 RKNN 模型信息

```bash
conda activate rknn

python scripts/07_print_rknn_model_info.py \
  --rknn outputs/rknn/cuadc-fp16.rknn
```

### 7. 部署到 RK3588

```bash
scp outputs/rknn/cuadc-fp16.rknn user@rk3588-host:/path/to/uav_system-rk3588/data/models/cuadc-fp16.rknn
```

板端配置 (`config/yolo.yaml`)：

```yaml
model_path: "../data/models/cuadc-fp16.rknn"
class_names: ["Target", "bucket", "class_2"]
selection_mode: "class"
target_class: "bucket"
conf_thres: 0.15
iou_thres: 0.45
```

重启 YOLO：

```bash
ssh user@rk3588-host
pkill -f "yolo_app.main"
cd ~/uav_system-rk3588
conda run --no-capture-output -n yolo \
  python3 -u -m yolo_app.main
```

## 环境要求

### 训练环境 (conda: uav)

```bash
conda activate uav
```

期望版本：
- Python 3.10.20
- torch 2.5.0+cu118
- ultralytics 8.3.163

### RKNN 转换环境 (conda: rknn)

如果没有 `rknn` 环境，需要创建：

```bash
conda create -n rknn python=3.10 -y
conda activate rknn
pip install rknn_toolkit2-*-cp310-*-linux_x86_64.whl
```

> ⚠️ 注意：必须安装 **x86_64 Linux** 版本的 rknn-toolkit2 wheel。

## 最终输出

`outputs/rknn/cuadc-fp16.rknn`

## 模型规格

| 参数 | 值 |
|------|-----|
| 输入尺寸 | 640×640 |
| 模型类型 | YOLO Detect |
| 输出格式 | single flat output |
| 输出 shape | (1, 4+nc, 8400) |
| nc=3 时 | (1, 7, 8400) |
| 最终格式 | .rknn (FP16) |
| target_platform | rk3588 |


## 手工标注

conda activate labelimg

labelImg datasets/cuadc2026/images/train datasets/cuadc2026/labels/train/classes.txt datasets/cuadc2026/labels/train

labelImg datasets/cuadc2026/images/val datasets/cuadc2026/labels/val/classes.txt datasets/cuadc2026/labels/val


labelImg datasets/gazebo_dataset/images/train datasets/gazebo_dataset/labels/train/classes.txt datasets/gazebo_dataset/labels/train

labelImg datasets/gazebo_dataset/images/val datasets/gazebo_dataset/labels/val/classes.txt datasets/gazebo_dataset/labels/val
