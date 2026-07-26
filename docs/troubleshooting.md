# 故障排查

---

## 1. INT8 检测不到任何目标

### 现象

```
scores min/max/mean = 0 / 0 / 0
detections = 0
```

### 原因

INT8 量化导致精度损失过大，模型输出崩塌。

### 处理

1. **切回 FP16**（推荐）
   ```bash
   python scripts/03_convert_rknn_fp16.py \
     --onnx outputs/onnx/cuadc.onnx \
     --output outputs/rknn/cuadc-fp16.rknn
   ```

2. 如果必须尝试 INT8：
   - 重新准备校准集，确保包含足够多样化的样本
   - 尝试不同的量化算法（normal / mmse / kl_divergence）
   - 尝试 W8A16 或 mixed precision

3. **不要**通过无限降低 `conf_thres` 来解决。如果 conf_thres < 0.01 才能检测到，说明模型本身有问题。

---

## 2. FP16 能检测但 FPS 低于预期

### 说明

FP16 稳定性更好，但 FPS 会低于 INT8。这是正常的权衡。

### 处理

- 优先保证识别和投放流程正确
- 如果 FPS 是硬需求，考虑：
  - 使用更小的模型（yolo11n 已经是最小）
  - 降低输入分辨率（如 320×320）
  - 优化板端后处理代码

---

## 3. 类别不匹配

### 现象

模型检测到的类别 ID 和预期不符。

### 原因

训练 `data.yaml` 的 `names` 顺序 ≠ 板端 `config/yolo.yaml` 的 `class_names` 顺序。

### 检查

训练端 `data.yaml`:
```yaml
names:
  0: Target
  1: bucket
  2: class_2
```

板端 `config/yolo.yaml`:
```yaml
class_names: ["Target", "bucket", "class_2"]
```

两者顺序必须一致。如果修改了一端，另一端必须同步。

---

## 4. 输出 shape 不对

### 期望

```
nc=3 时: output shape = (1, 7, 8400)
即 (1, 4 + nc, 8400)
```

### 检查项

1. **是否是 detect 模型？**
   ```python
   # 确认不是 segment / pose / classify 模型
   from ultralytics import YOLO
   model = YOLO("best.pt")
   print(model.task)  # 应该是 "detect"
   ```

2. **export 时 dynamic=False?**
   ```bash
   # 确认导出时 dynamic=False
   python scripts/02_export_onnx.py ... # 默认 dynamic=False
   ```

3. **imgsz=640?**
   ```bash
   python scripts/02_export_onnx.py --imgsz 640 ...
   ```

4. **类别数正确？**
   ```bash
   python scripts/06_test_onnx_output.py --nc 3 ...
   ```

---

## 5. RKNN 转换失败

### 现象

```
rknn.load_onnx() 返回非 0
或
rknn.build() 报错
```

### 检查

1. **是否在 x86_64 Linux 上运行？**
   - RKNN-Toolkit2 转换必须在 x86_64 上执行
   - 不能用 ARM 板上的 rknn-toolkit-lite2 做转换

2. **ONNX opset 版本是否兼容？**
   - 推荐 opset=12
   - 如果某些算子不支持，尝试 opset=11

3. **模型是否有不支持的操作？**
   - 简化后的模型更兼容
   - 确保 `simplify=True`

---

## 6. 环境问题

### 训练/导出时 ultralytics 报错

```bash
conda activate uav
python scripts/00_check_env.py
```

确保：
- Python 3.10.20
- torch 2.5.0+cu118
- ultralytics 8.3.163
- opencv-python 已安装

### RKNN 转换时 ImportError

```bash
conda activate rknn
python -c "from rknn.api import RKNN"
```

如果失败：
```bash
conda create -n rknn python=3.10 -y
conda activate rknn
pip install rknn_toolkit2-*-cp310-*-linux_x86_64.whl
```

---

## 7. ONNX 推理输出全 0

### 检查

```bash
python scripts/06_test_onnx_output.py \
  --onnx outputs/onnx/cuadc.onnx \
  --image /path/to/test_image.jpg
```

如果 ONNX 输出正常（有非零值）但 RKNN 输出全 0，说明转换有问题。

如果 ONNX 输出也是全 0，说明导出或训练有问题。
