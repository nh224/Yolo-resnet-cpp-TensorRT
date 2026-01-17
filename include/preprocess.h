#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// 定义预处理模式
enum PreprocessMode {
    MODE_LETTERBOX = 0,   // YOLO 风格 (保持宽高比 + 填充)
    MODE_STRETCH = 1,     // ResNet 拉伸风格 (直接缩放)
    MODE_CENTER_CROP = 2  // ImageNet 标准风格 (Resize + CenterCrop)
};

// 初始化与释放
void cuda_preprocess_init(int max_image_size);
void cuda_preprocess_destroy();

// 预处理核心函数 (根据模式自动选择专用核函数)
void cuda_preprocess(
    uint8_t* src,           // Host 端源图像数据 (BGR)
    int src_width,          // 源图像宽
    int src_height,         // 源图像高
    float* dst,             // Device 端目标 Buffer (NCHW, RGB)
    int dst_width,          // 目标宽
    int dst_height,         // 目标高
    cudaStream_t stream,    // CUDA 流
    const float* mean,      // 均值 (ResNet 使用)
    const float* std,       // 标准差 (ResNet 使用)
    PreprocessMode mode     // 模式
);

// Mask 解码函数
void cuda_decode_masks(
    float* mask_coeff_device,       // GPU: [N, mask_dim]
    float* mask_proto_device,       // GPU: [1, mask_dim, proto_h, proto_w]
    unsigned char* mask_out_device, // GPU: [N, out_h, out_w]
    int num_detections,
    int mask_dim,
    int proto_h,
    int proto_w,
    int out_h,
    int out_w,
    cudaStream_t stream
);