# YOLO ResNet C++ TensorRT

## 📜 引用

YOLO resnet C++ TensorRT 项目是一个高性能目标检测，图像分类解决方案，采用C++实现，并使用NVIDIA TensorRT进行优化。该项目利用 YOLOv8 v11 resnet 模型实现快速准确的目标检测与图像分类，并借助 TensorRT 最大程度地提高推理效率和性能。

---

## 📢 原作者

主要特点：
- 模型转换：将 ONNX 模型转换为 TensorRT 引擎文件以加速推理。
- 视频推理：高效地对视频文件进行目标检测。
- 图像推理：对单张图像执行目标检测。
- 高效：针对使用 NVIDIA GPU 的实时目标检测进行了优化。
- 使用 CUDA 进行预处理：启用 CUDA 的预处理，以加快输入处理速度。

https://github.com/nh224/Yolo-resnet-cpp-TensorRT/raw/main/asset/Bench_YOLO_V11.JPG

## 🛠️ 设置

### 先决条件
- CMake（版本 3.18 或更高版本）
- TensorRT（V8.6.1.6：针对使用 YOLOv11 的优化推理。）
- CUDA 工具包（V11.7：用于 GPU 加速）
- OpenCV（V4.10.0：用于图像和视频处理）
- NVIDIA GPU（计算能力7.5或更高）

### 安装
1. 克隆仓库：
```bash
git clone https://github.com/nh224/Yolo-resnet-cpp-TensorRT.git
cd YOLOv11-TensorRT
```

2. 更新 CMakeLists.txt 中的 TensorRT 和 OpenCV 路径：
```cmake
set(TENSORRT_PATH "F:/Program Files/TensorRT-8.6.1.6")  # Adjust this to your path
```

## 原作者功能

### 1. tensorrt版本兼容
```cpp
#if NV_TENSORRT_MAJOR < 10
    // For TensorRT versions less than 10, use getBindingDimensions
    input_h = engine->getBindingDimensions(0).d[2];
    input_w = engine->getBindingDimensions(0).d[3];
    detection_attribute_size = engine->getBindingDimensions(1).d[1];
    num_detections = engine->getBindingDimensions(1).d[2];
#else
    // For TensorRT versions 10 and above, use getTensorShape with tensor names
    auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
    
    auto output_dims = engine->getTensorShape(engine->getIOTensorName(1));
    detection_attribute_size = output_dims.d[1];
    num_detections = output_dims.d[2];
#endif
```

### 2. onnx转换为tensorrt
构建项目：
```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### 3. 对照片和视频推理

## 🚀 用法

### 将 Yolov11 转换为 ONNX 模型
```python
from ultralytics import YOLO
# Load the YOLO model
model = YOLO("yolo11s.pt")
#Export the model to ONNX format
export_path = model.export(format="onnx")
```

### 将 ONNX 模型转换为 TensorRT 引擎
要将 ONNX 模型转换为 TensorRT 引擎文件，请使用以下命令：
```bash
./YOLOv11TRT convert path_to_your_model.onnx path_to_your_engine.engine
```
- path_to_your_model.onnx：ONNX 模型文件的路径。
- path_to_your_engine.engine: TensorRT 引擎文件保存的路径。

### 对视频进行推理
要对视频进行推理，请使用以下命令：
```bash
./YOLOv11TRT infer_video path_to_your_video.mp4 path_to_your_engine.engine
```
- path_to_your_video.mp4：输入视频文件的路径。
- path_to_your_engine.engine：TensorRT 引擎文件的路径。

### 对照片进行推理
对图像运行推理 要对图像运行推理，请使用以下命令：
```bash
./YOLOv11TRT infer_image path_to_your_image.jpg path_to_your_engine.engine
```
- path_to_your_image.jpg：输入图像文件的路径。
- path_to_your_engine.engine：TensorRT 引擎文件的路径。

## 我的添加

### 1. 转换为tensorrt 兼容yolov8 v11 resnet
```bash
./YOLOv11TRT convert path_to_your_model.onnx path_to_your_engine.engine
```

### 2. 推理支持照片视频yolov8 v11 resnet
设置预处理参数并且使用cuda加速预处理
```cpp
void Wrapper::infer(const cv::Mat& input, std::vector<Object>& objects, int& cls_id, float& cls_score) {
    // 1. 预处理参数
    float mean[3], std[3];
    PreprocessMode p_mode;

    if (config_.type == ModelType::RESNET_CLS) {
        mean[0] = 0.485f; mean[1] = 0.456f; mean[2] = 0.406f;
        std[0]  = 0.229f; std[1]  = 0.224f; std[2]  = 0.225f;
        p_mode = MODE_STRETCH;
    } else {
        mean[0] = 0.0f; mean[1] = 0.0f; mean[2] = 0.0f;
        std[0]  = 1.0f; std[1]  = 1.0f; std[2]  = 1.0f;
        p_mode = MODE_LETTERBOX;
    }

    // 2. 执行预处理
    cuda_preprocess(
        input.data, input.cols, input.rows, 
        (float*)buffers[0], input_w, input_h, 
        stream, mean, std, p_mode
    );
```

后处理如果是yolo就做NMS如果是resnet就做softmax.
```cpp
// 3. 后处理
    if (config_.type == ModelType::YOLO_DETECT) {
        float scale = std::min((float)input_h / input.rows, (float)input_w / input.cols);
        postprocess_yolo(output_buffer_host, output_size, objects, scale, input.cols, input.rows);
    } 
    else {
        postprocess_resnet(output_buffer_host, output_size, cls_id, cls_score);
    }
```

### 编译后如何使用
```bash
./YOLOv11TRT infer_image resnet path_to_your_image.jpg path_to_your_engine.engine

./YOLOv11TRT infer_video resnet path_to_your_video.mp4 path_to_your_engine.engine

./YOLOv11TRT infer_image yolo path_to_your_image.jpg path_to_your_engine.engine

./YOLOv11TRT infer_video yolo path_to_your_video.mp4 path_to_your_engine.engine
```

## ⚙️ 配置

### CMake 配置
如果 TensorRT 和 OpenCV 安装在非默认位置，请在 CMakeLists.txt 文件中更新它们的路径：

设置 TensorRT 安装路径
```cmake
#Define the path to TensorRT installation
set(TENSORRT_PATH "F:/Program Files/TensorRT-8.6.1.6")  # Update this to the actual path for TensorRT
```
确保路径指向 TensorRT 的安装目录。

### 故障排除
找不到 nvinfer.lib：请确保 TensorRT 已正确安装，并且 nvinfer.lib 位于指定路径中。更新 CMakeLists.txt 文件，添加 TensorRT 库的正确路径。
链接器错误：请验证所有依赖项（OpenCV、CUDA、TensorRT）是否已正确安装，以及它们的路径是否已在 CMakeLists.txt 中正确设置。
运行时错误：请确保您的系统已安装正确的 CUDA 驱动程序，并且 TensorRT 运行时库可访问。将 TensorRT 的 bin 目录添加到系统 PATH 环境变量中。

## 📜 引用
我的代码是基于作者https://github.com/hamdiboukamcha/Yolo-V11-cpp-TensorRT
```
@misc{boukamcha2024yolov11,
    author = {Hamdi Boukamcha},
    title = {Yolo-V11-cpp-TensorRT},
    year = {2024},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/hamdiboukamcha/Yolo-V11-cpp-TensorRT/}},
}
```