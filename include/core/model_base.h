#pragma once
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include <vector>
#include <string>

// 抽象基类 - 所有模型的基础
class ModelBase {
public:
    ModelBase(const std::string& engine_path, bool is_segmentation = false);
    virtual ~ModelBase();

    // 纯虚函数 - 子类必须实现
    virtual void infer(const cv::Mat& input) = 0;

protected:
    // TensorRT 组件
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t stream = nullptr;

    // 内存缓冲区
    // 分割模型：0: Input, 1: Mask Proto, 2: BBox Output
    // 普通模型：0: Input, 1: Output
    std::vector<void*> gpu_buffers;

    // Pinned Memory (CPU端)
    float* cpu_output_buffer = nullptr;
    float* cpu_mask_buffer = nullptr;

    // Tensor 名称
    std::string input_tensor_name;
    std::vector<std::string> output_tensor_names;

    // 输入维度
    int input_w = 0;
    int input_h = 0;
    int input_size = 0;

    // 输出维度
    std::vector<int> output_sizes;
    int output_size = 0;  // 主输出大小（兼容旧代码）

    // YOLO 特有维度 (e.g., 84, 8400)
    int detection_attribute_size = 0;
    int num_detections = 0;

    // 是否为分割模型
    bool is_segmentation_ = false;

    // 初始化 TensorRT 上下文
    void init_context(const std::string& engine_path);

    // 获取分割模型的 mask 维度 (仅分割模型使用)
    void get_mask_dims(int& coeff_len, int& proto_h, int& proto_w) const;
};
