/**
 * @file postprocess.h
 * @brief CUDA 加速的后处理函数声明
 *
 * 提供 YOLO 检测、ResNet 分类、YOLO 分割的 GPU 后处理接口
 */

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include "types.h"

// ============================================================================
// YOLO 检测后处理（GPU 加速）
// ============================================================================
/**
 * @brief YOLO 检测后处理 - GPU 版本
 *
 * @param d_output           GPU: 原始输出 [attr_size, num_detections]
 * @param detection_attr_size 检测属性维度 (4 bbox + num_classes + mask_coeff)
 * @param num_detections     检测框数量 (如 8400)
 * @param num_classes        类别数量 (COCO 80)
 * @param conf_thres         置信度阈值
 * @param img_w              原图宽度
 * @param img_h              原图高度
 * @param input_w            模型输入宽度 (640)
 * @param input_h            模型输入高度 (640)
 * @param output             输出检测结果
 * @param nms_thres          NMS IoU 阈值
 * @param stream             CUDA 流
 */
void cuda_postprocess_yolo(
    float* d_output,
    int detection_attr_size,
    int num_detections,
    int num_classes,
    float conf_thres,
    int img_w, int img_h,
    int input_w, int input_h,
    std::vector<Detection>& output,
    float nms_thres,
    cudaStream_t stream
);

// ============================================================================
// ResNet 分类后处理（GPU 加速）
// ============================================================================
/**
 * @brief ResNet 分类后处理 - GPU 版本
 *
 * @param d_output   GPU: logits [num_classes]
 * @param num_classes 类别数量 (ImageNet 1000)
 * @param cls_id     输出: 最佳类别 ID
 * @param cls_score  输出: 最佳类别概率
 * @param stream     CUDA 流
 */
void cuda_postprocess_resnet(
    float* d_output,
    int num_classes,
    int& cls_id,
    float& cls_score,
    cudaStream_t stream
);

// ============================================================================
// YOLO 分割后处理（GPU 加速）
// ============================================================================
/**
 * @brief YOLO 实例分割后处理 - GPU 版本
 *
 * @param d_bbox_output     GPU: bbox 输出 [attr_size, num_detections]
 * @param d_mask_proto      GPU: mask 原型 [mask_coeff_len, proto_h, proto_w]
 * @param detection_attr_size 检测属性维度
 * @param num_detections     检测框数量
 * @param num_classes        类别数量
 * @param mask_coeff_len     mask 系数维度 (32)
 * @param mask_proto_h       mask 原型高度 (160)
 * @param mask_proto_w       mask 原型宽度 (160)
 * @param conf_thres         置信度阈值
 * @param img_w              原图宽度
 * @param img_h              原图高度
 * @param input_w            模型输入宽度
 * @param input_h            模型输入高度
 * @param output             输出检测结果（含 mask）
 * @param nms_thres          NMS IoU 阈值
 * @param stream             CUDA 流
 */
void cuda_postprocess_yolo_seg(
    float* d_bbox_output,
    float* d_mask_proto,
    int detection_attr_size,
    int num_detections,
    int num_classes,
    int mask_coeff_len,
    int mask_proto_h,
    int mask_proto_w,
    float conf_thres,
    int img_w, int img_h,
    int input_w, int input_h,
    std::vector<DetectionWithMask>& output,
    float nms_thres,
    cudaStream_t stream
);

// ============================================================================
// 初始化与清理函数
// ============================================================================
/**
 * @brief 初始化后处理模块（预分配 GPU 内存）
 */
void cuda_postprocess_init();

/**
 * @brief 清理后处理模块（释放 GPU 内存）
 */
void cuda_postprocess_destroy();
