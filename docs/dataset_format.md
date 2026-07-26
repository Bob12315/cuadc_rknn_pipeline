# 数据集格式说明

## 目录结构

数据集应组织为如下结构：

```
cuadc_dataset/
├── images/
│   ├── train/          # 训练集图片
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── val/            # 验证集图片
│   │   ├── img101.jpg
│   │   └── ...
│   └── test/           # 测试集图片（可选）
│       └── ...
└── labels/
    ├── train/          # 训练集标签 (.txt)
    │   ├── img001.txt
    │   ├── img002.txt
    │   └── ...
    ├── val/            # 验证集标签
    │   └── ...
    └── test/           # 测试集标签（可选）
        └── ...
```

## 标签格式

每张图片对应一个同名的 `.txt` 标签文件，格式为 YOLO 格式：

```
<class_id> <x_center> <y_center> <width> <height>
```

所有坐标归一化到 [0, 1]：

- `class_id`：类别编号（0-based）
- `x_center`：bbox 中心 x / 图片宽度
- `y_center`：bbox 中心 y / 图片高度
- `width`：bbox 宽度 / 图片宽度
- `height`：bbox 高度 / 图片高度

示例：
```
0 0.5234 0.4123 0.0891 0.1256
1 0.3456 0.6789 0.0450 0.0678
```

## 类别顺序

**⚠️ 关键约束：类别顺序必须一致！**

训练 `data.yaml` 的 `names` 顺序必须等于板端 `config/yolo.yaml` 的 `class_names` 顺序：

```yaml
# data.yaml (训练端)
names:
  0: Target
  1: bucket
  2: class_2

# config/yolo.yaml (板端)
class_names: ["Target", "bucket", "class_2"]
```

如果把 `class_2` 改成 `danger_sign`，则**两端必须同步修改**。

## 配置文件

参考 `configs/cuadc_data_template.yaml`，修改 `path` 为你的实际数据集路径：

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
