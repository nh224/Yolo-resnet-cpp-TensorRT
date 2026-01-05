#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// 定义预处理模式
enum PreprocessMode {
    MODE_LETTERBOX = 0, // YOLO 风格
    MODE_STRETCH = 1    // ResNet 风格
};

// 初始化与释放
void cuda_preprocess_init(int max_image_size);
void cuda_preprocess_destroy();

// 预处理核心函数
void cuda_preprocess(
    uint8_t* src,           // Host 端源图像数据
    int src_width,          // 源图像宽
    int src_height,         // 源图像高
    float* dst,             // Device 端目标 Buffer
    int dst_width,          // 目标宽
    int dst_height,         // 目标高
    cudaStream_t stream,    // CUDA 流
    const float* mean,      // 均值
    const float* std,       // 标准差
    PreprocessMode mode     // 模式
);