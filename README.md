YOLO & ResNet C++ TensorRT
高性能目标检测与图像分类解决方案

本项目是一个基于 C++ 实现的高性能推理解决方案，利用 NVIDIA TensorRT 对模型进行极致优化。项目不仅支持 YOLOv8 / YOLOv11 进行快速准确的目标检测，还集成了 ResNet 模型用于图像分类。

Benchmark

📢 主要特点
多模型支持：支持 YOLOv8、YOLOv11（目标检测）及 ResNet（图像分类）。
模型转换：提供工具将 ONNX 模型转换为高效的 TensorRT Engine 文件。
CUDA 加速预处理：利用 CUDA 核函数进行图像预处理（Letterbox 或 Stretch），大幅提升输入处理速度。
多媒体推理：支持对 单张图像 和 视频文件 进行推理。
版本兼容性：代码适配了 TensorRT 8.x 及 10.x 版本 API 的差异。
🛠️ 环境准备 (Prerequisites)
在编译和运行本项目之前，请确保满足以下条件：

CMake: 版本 3.18 或更高
TensorRT:
推荐 v8.6.1.6 (针对 YOLOv11 优化)
兼容 v10.x (已做代码适配)
CUDA Toolkit: v11.7 (用于 GPU 加速)
OpenCV: v4.10.0 (用于图像读取和视频处理)
Hardware: NVIDIA GPU (计算能力 7.5 或更高)
🚀 安装与构建
1. 克隆仓库
Bash

git clone https://github.com/nh224/Yolo-resnet-cpp-TensorRT.git
cd YOLOv11-TensorRT
2. 配置 CMake
打开 CMakeLists.txt，根据你的实际安装路径修改 TensorRT 和 OpenCV 的路径：

cmake

set(TENSORRT_PATH "F:/Program Files/TensorRT-8.6.1.6")  # 请修改为您实际的 TensorRT 路径
3. 编译项目
Bash

mkdir build
cd build
cmake ..
make -j$(nproc)
🏃 使用指南
1. 导出 ONNX 模型
首先需要将 PyTorch 模型 (.pt) 导出为 ONNX 格式。

Python

from ultralytics import YOLO

# 加载模型 (YOLOv11 或 YOLOv8)
model = YOLO("yolo11s.pt")

# 导出为 ONNX
model.export(format="onnx")
2. 转换为 TensorRT Engine
使用编译好的可执行文件将 ONNX 转换为 Engine 文件。该命令兼容 YOLOv8/v11 和 ResNet。

Bash

./YOLOv11TRT convert path_to_your_model.onnx path_to_your_engine.engine
3. 执行推理
本项目支持通过命令行参数指定模型类型（yolo 或 resnet）以及输入类型（图像 or 视频）。

📷 图像推理
ResNet 分类:

Bash

./YOLOv11TRT infer_image resnet path_to_your_image.jpg path_to_your_engine.engine
YOLO 目标检测:

Bash

./YOLOv11TRT infer_image yolo path_to_your_image.jpg path_to_your_engine.engine
🎥 视频推理
ResNet 分类:

Bash

./YOLOv11TRT infer_video resnet path_to_your_video.mp4 path_to_your_engine.engine
YOLO 目标检测:

Bash

./YOLOv11TRT infer_video yolo path_to_your_video.mp4 path_to_your_engine.engine
🧠 技术实现细节
1. TensorRT 版本兼容性处理
代码自动检测 TensorRT 版本，以适配 getBindingDimensions (v8) 和 getTensorShape (v10+) 的 API 变更。

C++

#if NV_TENSORRT_MAJOR < 10
    // TensorRT < 10
    input_h = engine->getBindingDimensions(0).d[2];
    input_w = engine->getBindingDimensions(0).d[3];
    // ...
#else
    // TensorRT >= 10
    auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
    // ...
#endif
2. CUDA 预处理逻辑
根据模型类型选择不同的预处理策略，全部在 GPU 上完成以减少 CPU-GPU 内存拷贝。

ResNet: 使用 MODE_STRETCH，并应用 ImageNet 均值/方差归一化。
YOLO: 使用 MODE_LETTERBOX (保持宽高比填充)，归一化至 [0, 1]。
C++

void Wrapper::infer(const cv::Mat& input, std::vector<Object>& objects, int& cls_id, float& cls_score) {
    float mean[3], std[3];
    PreprocessMode p_mode;

    if (config_.type == ModelType::RESNET_CLS) {
        // ResNet 参数
        mean[0] = 0.485f; mean[1] = 0.456f; mean[2] = 0.406f;
        std[0]  = 0.229f; std[1]  = 0.224f; std[2]  = 0.225f;
        p_mode = MODE_STRETCH;
    } else {
        // YOLO 参数
        mean[0] = 0.0f; mean[1] = 0.0f; mean[2] = 0.0f;
        std[0]  = 1.0f; std[1]  = 1.0f; std[2]  = 1.0f;
        p_mode = MODE_LETTERBOX;
    }

    // CUDA 预处理调用
    cuda_preprocess(
        input.data, input.cols, input.rows, 
        (float*)buffers[0], input_w, input_h, 
        stream, mean, std, p_mode
    );
    
    // ... 推理与后处理 ...
}
3. 后处理分流
YOLO: 执行 NMS (非极大值抑制) 并映射回原图坐标。
ResNet: 执行 Softmax 获取分类 ID 和置信度。
⚙️ 故障排除 (Troubleshooting)
找不到 nvinfer.lib:
确保 CMakeLists.txt 中 TENSORRT_PATH 设置正确。
检查 TensorRT 是否已正确安装且与 CUDA 版本匹配。
链接器错误 (Linker Errors):
验证 OpenCV、CUDA 和 TensorRT 的库路径是否在系统环境变量或 CMake 配置中正确包含。
运行时错误 (Runtime Errors):
确保已安装最新的 NVIDIA 驱动程序。
将 TensorRT 的 lib 或 bin 目录添加到系统环境变量 PATH (Windows) 或 LD_LIBRARY_PATH (Linux) 中。
推理结果不正确:
检查导出 ONNX 时是否使用了正确的 opset 版本。
确认输入图像的预处理均值/方差与训练时一致。
📜 引用与致谢
本项目基于 Hamdi Boukamcha 的工作进行改进与扩展。

Original Author:

bibtex

@misc{boukamcha2024yolov11,
    author = {Hamdi Boukamcha},
    title = {Yolo-V11-cpp-TensorRT},
    year = {2024},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/hamdiboukamcha/Yolo-V11-cpp-TensorRT/}},
}





