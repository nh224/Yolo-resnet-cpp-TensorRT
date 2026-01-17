# YOLOv11 ResNet C++ TensorRT

<div align="center">

![TensorRT](https://img.shields.io/badge/TensorRT-8.6+-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.7+-green.svg)
![C++](https://img.shields.io/badge/C++-17-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

高性能 YOLOv11/v8 + ResNet C++ 推理框架，使用 NVIDIA TensorRT 加速

[功能](#功能) • [快速开始](#快速开始) • [使用文档](#使用文档) • [性能](#性能)

</div>

---

## 📋 目录

- [功能](#功能)
- [支持的模型](#支持的模型)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [使用方法](#使用方法)
- [性能测试](#性能测试)
- [常见问题](#常见问题)
- [许可证](#许可证)

---

## ✨ 功能

- **🚀 高性能推理**: 使用 TensorRT 优化，支持 FP16/FP32 精度
- **⚡ CUDA 加速预处理**: 完全在 GPU 上进行图像预处理
- **🎯 多模型支持**: YOLOv11/v8 检测、分割、ResNet 分类
- **📹 视频推理**: 支持视频文件实时推理
- **🔧 易于使用**: 简单的命令行接口

---

## 🎯 支持的模型

| 模型类型 | 支持架构 | 输入尺寸 | 说明 |
|---------|---------|---------|------|
| **目标检测** | YOLOv8, YOLOv11 | 640×640 | 检测并定位图像中的物体 |
| **实例分割** | YOLOv8-Seg, YOLOv11-Seg | 640×640 | 检测物体并返回像素级掩码 |
| **图像分类** | ResNet18/50, EfficientNet 等 | 224×224 | ImageNet 1000类分类 |

---

## 🛠️ 环境要求

### 硬件
- **GPU**: NVIDIA GPU (计算能力 7.5+)
- **显存**: 建议 4GB+

### 软件
| 依赖 | 版本要求 |
|------|---------|
| CMake | 3.18+ |
| CUDA | 11.7+ |
| TensorRT | 8.6.1+ |
| OpenCV | 4.0+ |

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-repo/Yolo-V11-cpp-TensorRT.git
cd Yolo-V11-cpp-TensorRT
```

### 2. 配置 TensorRT 路径

编辑 `CMakeLists.txt`：

```cmake
set(TENSORRT_PATH "/usr/local/TensorRT-8.6.1.6")  # 修改为你的路径
```

### 3. 编译

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

编译成功后生成 `YOLOv11TRT` 可执行文件。

---

## 📖 使用方法

### 命令格式

```bash
./YOLOv11TRT <mode> <model_type> <input_path> <engine_path> [options]
```

### 可用模式

| 模式 | 说明 |
|------|------|
| `convert` | ONNX → TensorRT 引擎 |
| `infer_image` | 图像推理 |
| `infer_video` | 视频推理 |
| `infer_segment` | 分割推理 |
| `benchmark` | 性能测试 |

### 使用示例

#### 1️⃣ 模型转换

```bash
# FP16 模式 (默认，速度快)
./YOLOv11TRT convert model.onnx model.engine

# FP32 模式 (精度最高)
./YOLOv11TRT convert model.onnx model.engine --fp32
```

#### 2️⃣ 图像分类 (ResNet)

```bash
./YOLOv11TRT infer_image resnet test_images/car.jpg resnet18.engine
```


#### 3️⃣ 目标检测 (YOLO)

```bash
./YOLOv11TRT infer_image yolo test_images/dog.jpg yolo11s.engine
```

#### 4️⃣ 实例分割 (YOLO-Seg)

```bash
./YOLOv11TRT infer_segment yolo-seg test_images/person.jpg yolov8s-seg.engine
```

#### 5️⃣ 视频推理

```bash
./YOLOv11TRT infer_video yolo video.mp4 yolo11s.engine
```

#### 6️⃣ 性能测试

```bash
# 默认 100 次
./YOLOv11TRT benchmark resnet test.jpg resnet18.engine

# 自定义测试次数
./YOLOv11TRT benchmark yolo test.jpg yolo11s.engine 80 1000
```

---

## ⚡ 性能

### 测试环境
- **GPU**: NVIDIA Jetson Orin
- **CUDA**: 11.7
- **TensorRT**: 8.6.1.6

### 性能数据

| 模型 | 输入尺寸 | FP16 延迟 | FP32 延迟 | FPS (FP16) |
|------|---------|----------|----------|-----------|
| YOLOv11s | 640×640 | ~5ms | ~8ms | ~200 |
| ResNet18 | 224×224 | ~2.5ms | ~3ms | ~400 |
| YOLOv8s-Seg | 640×640 | ~7ms | ~11ms | ~140 |


## 📂 项目结构

```
Yolo-V11-cpp-TensorRT/
├── include/              # 头文件
│   ├── core/            # TensorRT 引擎构建
│   ├── models/          # 模型封装
│   └── types.h          # 类型定义
├── src/                 # 源文件
│   ├── core/            # TensorRT 引擎实现
│   ├── models/          # 模型推理实现
│   ├── preprocess.cu    # CUDA 预处理
│   └── postprocess.cu   # CUDA 后处理
├── model_weights/       # 模型权重
├── test_images/         # 测试图像
├── outputs/             # 输出结果
├── CMakeLists.txt      # 构建配置
├── main.cpp             # 主程序
└── README.md           # 本文件
```

---

## ❓ 常见问题

### Q: 编译时找不到 TensorRT 头文件？

**A**: 检查 `CMakeLists.txt` 中的 `TENSORRT_PATH` 是否正确。

### Q: 运行时找不到共享库？

**A**: 添加 TensorRT lib 目录到 `LD_LIBRARY_PATH`：

```bash
export LD_LIBRARY_PATH=/usr/local/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH
```

### Q: 如何导出 ResNet ONNX 模型？

**A**: 参考 `export_resnet_onnx.py`：

```python
from torchvision import models
import torch

# 加载模型
model = models.resnet18(pretrained=True)
model.eval()

# 导出 ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "resnet18.onnx",
                  opset_version=17, export_params=True)
```

---

## 📜 引用

本项目基于以下开源项目：

- **原始项目**: [hamdiboukamcha/Yolo-V11-cpp-TensorRT](https://github.com/hamdiboukamcha/Yolo-V11-cpp-TensorRT)
- **YOLO**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

```bibtex
@misc{boukamcha2024yolov11,
    author = {Hamdi Boukamcha},
    title = {Yolo-V11-cpp-TensorRT},
    year = {2024},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/hamdiboukamcha/Yolo-V11-cpp-TensorRT/}},
}
```

---

## 📄 许可证

MIT License

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个 Star！**

</div>