# CUADC 模型训练与转换手册（YOLO → PT → ONNX → RKNN）

本仓库只负责训练端模型流水线：**数据标注/数据集 → YOLO Detect 训练产出 `.pt` → ONNX → RK3588 目标平台的 FP16 `.rknn`**。不包含 RK3588 板端程序、模型上传、NPU runtime、服务配置或推理部署。

## 当前项目与边界

- 默认模型：YOLO11n Detect；输入尺寸：`640×640`；ONNX：静态输入、`opset=12`。
- 默认转换产物：RK3588 FP16 RKNN；INT8 仅为可选实验，当前不作为交付模型。
- 训练、ONNX 导出和 RKNN 转换都在 **x86_64 Linux 训练主机**进行；本文不要求准备 RK3588 开发板。
- 当前仓库没有实际 `datasets/`、数据集 YAML、训练权重、`rknn` 环境或 Toolkit2 wheel。
- 本机的训练环境名是 `uav-dev`，而脚本默认名是 `uav`；调用总流水线时必须传入 `--train-env uav-dev`。

## 所需内容与当前状态

| 类别 | 必需内容 | 当前状态 / 验收方式 |
|---|---|---|
| 训练主机 | x86_64 Linux、Conda、网络、足够磁盘 | `uname -m` 必须为 `x86_64` |
| GPU（推荐） | NVIDIA 驱动、GPU、匹配 CUDA 的 PyTorch | `nvidia-smi` 和 `torch.cuda.is_available()` 正常 |
| 标注 | 图形桌面、LabelImg 环境、类别文件 | LabelImg 保存 YOLO `.txt` 标签 |
| 数据集 | train/val 图片与同名标签、实际 data.yaml | 每张图片都能对应一个 `.txt`；空标签用于负样本 |
| 训练环境 | Python 3.10、Torch、项目 requirements | `00_check_env.py` 显示训练/导出依赖正常 |
| 转换环境 | Python 3.10、官方 x86_64 CPython 3.10 RKNN-Toolkit2 | `from rknn.api import RKNN` 成功 |
| 交付产物 | `best.pt`、`.onnx`、FP16 `.rknn` | 三个文件均存在，ONNX 输出形状正确 |

## 官方链接

| 内容 | 链接 | 用途 |
|---|---|---|
| Conda | [环境管理文档](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) | 创建和激活环境 |
| PyTorch | [Start Locally](https://docs.pytorch.org/get-started/locally/) | 根据实际驱动/CUDA 选择 Torch 安装命令 |
| Ultralytics 数据集 | [Detect Datasets](https://docs.ultralytics.com/datasets/detect/) | YOLO data.yaml 与标签格式 |
| Ultralytics 训练 | [Train](https://docs.ultralytics.com/modes/train/) | 训练参数参考 |
| Ultralytics 导出 | [Export](https://docs.ultralytics.com/modes/export/) | ONNX 导出参数参考 |
| LabelImg | [官方仓库](https://github.com/HumanSignal/labelImg) | 人工图像标注 |
| RKNN-Toolkit2 | [Rockchip 官方仓库](https://github.com/airockchip/rknn-toolkit2) | 获取官方转换包和文档 |
| RKNN Model Zoo | [Rockchip 官方示例](https://github.com/airockchip/rknn_model_zoo) | 参考 RK3588 目标平台的模型转换兼容性 |

## 1. 创建训练环境

当前 `uav-dev` 仅有 Python 3.10，Torch、Ultralytics、ONNX、ONNX Runtime 和 OpenCV 未安装。

```bash
conda create -n uav-dev python=3.10 -y  # 仅当环境不存在时执行
conda activate uav-dev

cd /home/level6/cuadc_rknn_pipeline
```

若终端已经激活其他 `venv`，先执行 `deactivate`；确认 `which python` 指向 `.../envs/uav-dev/bin/python`，避免包装错环境。

### PyTorch CUDA / 镜像源选择

先查看显卡和驱动：

```bash
nvidia-smi
```

不要只按显卡代数选择 wheel，**NVIDIA 驱动版本必须支持所选 CUDA 运行时**。本机是 Quadro RTX 3000（非 50 系）、驱动 535、最高支持 CUDA 12.2，因此项目固定使用 `torch 2.5.0 + cu118`。下面的阿里云 PyTorch wheel 镜像目录当前提供 `cpu`、`cu118`、`cu121`、`cu124`、`cu126` 和 `cu128`。

| 适用情况 | 建议 CUDA wheel | 阿里云镜像地址 | 说明 |
|---|---|---|---|
| RTX 20 / 30 / 40 或其他非 50 系；驱动支持 CUDA 11.8 | `cu118` | [cu118](https://mirrors.aliyun.com/pytorch-wheels/cu118/) | 本项目与本机使用这一档，兼容性优先 |
| 非 50 系；驱动支持 CUDA 12.1 | `cu121` | [cu121](https://mirrors.aliyun.com/pytorch-wheels/cu121/) | 需要较新的驱动；只在项目/驱动确有需要时使用 |
| 非 50 系；驱动支持 CUDA 12.4 | `cu124` | [cu124](https://mirrors.aliyun.com/pytorch-wheels/cu124/) | 需同步选择相匹配的 Torch 版本 |
| RTX 50 系；驱动满足 CUDA 12.8 要求 | `cu128` | [cu128](https://mirrors.aliyun.com/pytorch-wheels/cu128/) | 先在 PyTorch 官方选择器确认当前推荐版本与最低驱动 |
| 仅 CPU 或无 NVIDIA GPU | `cpu` | [cpu](https://mirrors.aliyun.com/pytorch-wheels/cpu/) | 可运行调试和导出，正式训练速度很慢 |

阿里云镜像使用 `-f`（find-links）提供 PyTorch CUDA wheel，PyPI 仍用于解析普通依赖。针对本项目的非 50 系 NVIDIA 显卡，执行：

```bash
python -m pip install \
  torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
  -f https://mirrors.aliyun.com/pytorch-wheels/cu118
```

其他 CUDA 目录可按相同形式替换，例如 RTX 50 系采用 `-f https://mirrors.aliyun.com/pytorch-wheels/cu128`，但 **Torch / TorchVision / TorchAudio 版本组合必须从 [PyTorch 官方安装页](https://docs.pytorch.org/get-started/locally/) 确认后再固定**，不要将本项目的 2.5.0 组合直接套用于新 CUDA 版本。

安装 Torch 后再安装项目依赖并验证：

```bash
python -m pip install -r requirements.txt
python scripts/00_check_env.py
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2. LabelImg 标注环境

已安装的环境：`/home/level6/miniforge3/envs/labelimg`。类别模板为 [`configs/labelimg_classes.txt`](../configs/labelimg_classes.txt)，每一行对应一个从 0 开始的类别 ID。开始标注前按实际项目修改它。

```bash
cd /home/level6/cuadc_rknn_pipeline
conda activate labelimg
labelImg /绝对路径/cuadc_dataset/images/train configs/labelimg_classes.txt
```

如果 shell 的 PATH 被其他 Python 环境覆盖，使用已验证的绝对路径：

```bash
/home/level6/miniforge3/envs/labelimg/bin/labelImg \
  /绝对路径/cuadc_dataset/images/train \
  /home/level6/cuadc_rknn_pipeline/configs/labelimg_classes.txt
```

在 LabelImg 中选择 **YOLO** 格式，并用 “Change Save Dir” 将标签保存到 `/绝对路径/cuadc_dataset/labels/train`。验证集图片和标签使用同样方法放入 `images/val`、`labels/val`。

## 3. 数据集结构与校验

```text
cuadc_dataset/
├── images/
│   ├── train/  # 图片
│   └── val/
├── labels/
│   ├── train/  # 同名 .txt 标签
│   └── val/
└── cuadc.yaml
```

标签每一行为：

```text
<class_id> <x_center> <y_center> <width> <height>
```

后四项为 `[0,1]` 范围内的归一化坐标。建议按视频或采集批次拆分 train/val，避免连续相邻帧同时进入两个集合导致验证结果虚高。

标注后执行此检查（替换真实路径）：

```bash
python - <<'PY'
from pathlib import Path
root = Path('/绝对路径/cuadc_dataset')
for split in ('train', 'val'):
    images = [p for p in (root / 'images' / split).iterdir()
              if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
    missing = [p.name for p in images if not (root / 'labels' / split / f'{p.stem}.txt').is_file()]
    print(f'{split}: images={len(images)}, missing_labels={len(missing)}')
    if missing:
        print('examples:', missing[:10])
PY
```

## 4. 创建 data.yaml

将下列内容保存为 `<dataset_root>/cuadc.yaml`。`path` 和类别名必须改为实际值；类别顺序必须等于 `labelimg_classes.txt` 的行顺序。

```yaml
path: /绝对路径/cuadc_dataset
train: images/train
val: images/val
# test: images/test
names:
  0: Target
  1: bucket
  2: class_2
nc: 3
```

## 5. 创建 RKNN 转换环境

转换需要独立的 `rknn` 环境。仅安装与 **Linux x86_64、Python 3.10（cp310）** 相匹配的官方 wheel；不要安装 ARM/aarch64 或其他 Python ABI 的 wheel。

```bash
cd /home/level6
git clone --depth 1 https://github.com/airockchip/rknn-toolkit2.git

conda create -n rknn python=3.10 -y
conda activate rknn

# 先找出当前官方仓库实际提供的 cp310/x86_64 wheel 与配套依赖文件。
find /home/level6/rknn-toolkit2 -type f \( -name '*cp310*linux*x86_64*.whl' -o -name 'requirements*cp310*.txt' \) -print

# 以下两行的“实际路径”替换为上一步找到的文件路径。
python -m pip install -r /实际路径/requirements_cp310-*.txt
python -m pip install /实际路径/rknn_toolkit2-*-cp310-*-linux_x86_64.whl
python -c "from rknn.api import RKNN; print('RKNN-Toolkit2 OK')"
```

训练环境和 RKNN 环境必须隔离，以避免 ONNX、Numpy 等依赖发生冲突。

## 6. 训练、导出和转换

先用 1 epoch 冒烟运行确认数据路径、标签和 GPU，再执行正式训练。显存不足时降低 `--batch`；当前项目输入尺寸应保持 `640`。

```bash
cd /home/level6/cuadc_rknn_pipeline

# 冒烟运行
python scripts/00_train_pipeline.py \
  --data /绝对路径/cuadc_dataset/cuadc.yaml \
  --name cuadc_smoke \
  --train-env uav-dev --rknn-env rknn \
  --imgsz 640 --epochs 1 --batch 4 --device 0 --workers 4

# 正式训练 → ONNX → FP16 RKNN
python scripts/00_train_pipeline.py \
  --data /绝对路径/cuadc_dataset/cuadc.yaml \
  --name cuadc \
  --train-env uav-dev --rknn-env rknn \
  --imgsz 640 --epochs 150 --batch 16 --device 0 --workers 4
```

正式产物路径：

```text
outputs/train/cuadc/weights/best.pt
outputs/onnx/cuadc.onnx
outputs/rknn/cuadc-fp16.rknn
```

## 7. 模型产物验收

```bash
cd /home/level6/cuadc_rknn_pipeline

# 检查训练环境
/home/level6/miniforge3/envs/uav-dev/bin/python scripts/00_check_env.py

# 检查 ONNX 输出，替换验证集图片路径
/home/level6/miniforge3/envs/uav-dev/bin/python scripts/06_test_onnx_output.py \
  --onnx outputs/onnx/cuadc.onnx \
  --image /绝对路径/cuadc_dataset/images/val/一张图片.jpg

# 确认产物存在
ls -lh outputs/train/cuadc/weights/best.pt outputs/onnx/cuadc.onnx outputs/rknn/cuadc-fp16.rknn
```

对于 `nc=3`，ONNX 输出应为 `(1, 7, 8400)`，即 `(1, 4+nc, 8400)`。最终交付仅包含上列 `.pt`、`.onnx` 和 `-fp16.rknn` 三个模型文件及对应的 data.yaml、类别列表和版本记录。

## 可直接交给 AI 的提示词

```text
你负责本仓库的训练端模型产出，不负责 RK3588 板端部署、上传模型、NPU runtime、服务配置或推理程序。项目目录为 /home/level6/cuadc_rknn_pipeline；目标是完成：YOLO Detect 训练 → best.pt → 固定 640×640 ONNX（opset 12）→ RK3588 FP16 RKNN。

使用 Conda 环境：训练/ONNX 使用 uav-dev；RKNN 转换使用 rknn；人工标注使用 labelimg。脚本默认训练环境名为 uav，所以总流水线必须传 `--train-env uav-dev`。先检查 x86_64 Linux、GPU/CUDA、三套环境、实际数据集 YAML、类别文件以及官方 cp310 Linux x86_64 RKNN Toolkit2 wheel。不得猜测路径或类别。

类别文件 configs/labelimg_classes.txt 的行顺序、data.yaml 的 names 顺序必须完全一致。LabelImg 选择 YOLO 格式，将同名 txt 标签写入 labels/train 或 labels/val。数据集 YAML 必须用绝对 path，且 train/val 图片与标签一一对应。先运行 1 epoch 冒烟流程；通过后运行：

python scripts/00_train_pipeline.py --data <dataset_yaml> --name <name> --train-env uav-dev --rknn-env rknn --imgsz 640 --epochs 150 --batch <按显存设置> --device 0 --workers 4

输出并验证：outputs/train/<name>/weights/best.pt、outputs/onnx/<name>.onnx、outputs/rknn/<name>-fp16.rknn。转换前检查 ONNX 是 detect 模型；类别数 nc 时输出形状应为 (1, 4+nc, 8400)。默认只交付 FP16 RKNN；INT8 不作为主产物。最后报告实际环境版本、data.yaml、类别映射、三个模型文件绝对路径和验证结果。
```
