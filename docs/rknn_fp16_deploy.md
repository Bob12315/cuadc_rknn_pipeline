# RK3588 FP16 部署说明

## 概述

FP16 RKNN 模型是当前项目的**默认部署模型**。

- 稳定性好，识别结果可靠
- 之前 INT8 出现过 class score 全 0 / detections=0 的问题
- 优先保证识别和投放流程

---

## 部署步骤

### 1. 上传模型到 RK3588

```bash
scp outputs/rknn/cuadc-fp16.rknn user@rk3588-host:/path/to/uav_system-rk3588/data/models/cuadc-fp16.rknn
```

### 2. 板端配置

编辑 RK3588 上的 `config/yolo.yaml`：

```yaml
model_path: "../data/models/cuadc-fp16.rknn"

class_names: ["Target", "bucket", "class_2"]

selection_mode: "class"
target_class: "bucket"

conf_thres: 0.15
iou_thres: 0.45
```

### 3. 重启 YOLO 服务

```bash
ssh user@rk3588-host

# 停止旧进程
pkill -f "yolo_app.main"

# 启动
cd ~/uav_system-rk3588

conda run --no-capture-output -n yolo \
  python3 -u -m yolo_app.main
```

---

## 模型规格

| 参数 | 值 |
|------|-----|
| 输入尺寸 | 640×640 |
| 模型类型 | YOLO Detect |
| 输出格式 | single flat output |
| 输出 shape (nc=3) | (1, 7, 8400) |
| 格式 | .rknn (FP16) |
| target_platform | rk3588 |
| mean_values | [0, 0, 0] |
| std_values | [255, 255, 255] |

---

## FP16 vs INT8

| 特性 | FP16 | INT8 |
|------|------|------|
| 状态 | **默认部署** | 可选实验 |
| 稳定性 | 高 | 之前出现过 score 全 0 |
| 速度 | 中等 | 更快 |
| 精度 | 高 | 可能降低 |
| 推荐 | ✅ 是 | ⚠️ 需充分验证 |

---

## 注意事项

1. **类别顺序必须一致**：训练 data.yaml 的 names = 板端 config/yolo.yaml 的 class_names
2. **不要无限降低 conf_thres** 来补偿模型问题
3. 如果模型不识别，先检查：
   - ONNX 输出 shape 是否为 (1, 4+nc, 8400)
   - 是否是 detect 模型（不是 segment/pose）
   - 板端 class_names 顺序是否和训练一致
