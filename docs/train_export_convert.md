# 训练 / 导出 / 转换 完整操作指南

## 一、环境配置

### 训练环境 (conda: uav)

```bash
conda activate uav
```

检查版本：

```bash
python -V
# Python 3.10.20

python -c "import torch; print(torch.__version__)"
# 2.5.0+cu118

python -c "import ultralytics; print(ultralytics.__version__)"
# 8.3.163
```

运行环境检查脚本：

```bash
python scripts/00_check_env.py
```

### RKNN 转换环境 (conda: rknn)

如果还没有 `rknn` 环境：

```bash
conda create -n rknn python=3.10 -y
conda activate rknn
```

安装 x86_64 Linux 版本的 rknn-toolkit2：

```bash
pip install rknn_toolkit2-*-cp310-*-linux_x86_64.whl
```

> ⚠️ **重要**：
> - 必须安装 **x86_64 Linux** 版本的 wheel
> - **不要**在 RK3588 ARM 板上运行 RKNN-Toolkit2 转换
> - RK3588 只负责运行 `.rknn` 模型，不负责转换

---

## 二、训练

### 准备数据集

1. 按照 `docs/dataset_format.md` 组织数据集目录
2. 编辑 `configs/cuadc_data_template.yaml`，填写正确的 `path`

### 运行训练

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

### 训练输出

```
outputs/train/cuadc_yolo11n_640/
├── weights/
│   ├── best.pt      ← 使用这个
│   └── last.pt
├── results.csv
└── ...
```

---

## 三、导出 ONNX

```bash
conda activate uav

python scripts/02_export_onnx.py \
  --weights outputs/train/cuadc_yolo11n_640/weights/best.pt \
  --imgsz 640 \
  --opset 12 \
  --output outputs/onnx/cuadc.onnx
```

### 导出参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| format | onnx | 固定 |
| imgsz | 640 | 固定 640×640 |
| opset | 12 | ONNX opset 版本 |
| simplify | True | 简化模型图 |
| dynamic | False | 不导出动态 batch |

### 检测 ONNX 输出

```bash
python scripts/06_test_onnx_output.py \
  --onnx outputs/onnx/cuadc.onnx \
  --image /path/to/any_test_image.jpg
```

期望输出：
```
output shape = (1, 7, 8400)    # nc=3 时，(1, 4+nc, 8400)
```

---

## 四、转换 FP16 RKNN

```bash
conda activate rknn

python scripts/03_convert_rknn_fp16.py \
  --onnx outputs/onnx/cuadc.onnx \
  --output outputs/rknn/cuadc-fp16.rknn
```

### 转换参数

| 参数 | 值 |
|------|-----|
| target_platform | rk3588 |
| do_quantization | False (FP16) |
| mean_values | [[0, 0, 0]] |
| std_values | [[255, 255, 255]] |

---

## 五、（可选）转换 INT8 RKNN

> ⚠️ INT8 是可选实验流程，默认不部署。

```bash
# 先生成校准集
python scripts/05_make_calib_dataset.py \
  --input-dir /path/to/raw_calib_images \
  --output-dir outputs/calib/images_640 \
  --dataset-txt outputs/calib/dataset.txt \
  --imgsz 640

# 再转换 INT8
conda activate rknn

python scripts/04_convert_rknn_int8_optional.py \
  --onnx outputs/onnx/cuadc.onnx \
  --dataset outputs/calib/dataset.txt \
  --output outputs/rknn/cuadc-int8.rknn
```

---

## 六、部署

详见 `docs/rknn_fp16_deploy.md`。
