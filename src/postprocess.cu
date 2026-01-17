/**
 * @file postprocess.cu
 * @brief CUDA 加速的后处理实现
 *
 * 包含：
 * - YOLO 检测后处理（置信度过滤 + NMS + 坐标映射）
 * - ResNet 分类后处理（Softmax）
 * - YOLO 分割后处理（置信度过滤 + NMS + Mask 解码）
 */

#include "postprocess.h"
#include "preprocess.h"
#include "cuda_utils.h"
#include <cmath>
#include <device_launch_parameters.h>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <opencv2/opencv.hpp>

// ============================================================================
// 常量定义
// ============================================================================
constexpr int MAX_DETECTIONS = 8400;        // YOLO 最大检测数
constexpr int THREADS_PER_BLOCK = 256;

// ============================================================================
// 预分配 GPU 缓冲区
// ============================================================================
static float* d_filtered_boxes = nullptr;      // [MAX_DETS, 4]
static int* d_filtered_class_ids = nullptr;    // [MAX_DETS]
static float* d_filtered_conf = nullptr;       // [MAX_DETS]
static int* d_filtered_indices = nullptr;      // [MAX_DETS] - 原始索引
static int* d_filtered_count = nullptr;        // [1]
static float* d_iou_matrix = nullptr;          // [MAX_DETS, MAX_DETS]
static int* d_suppressed = nullptr;            // [MAX_DETS]
static int* d_keep_indices = nullptr;          // [MAX_DETS]
static int* d_keep_count = nullptr;            // [1]

static float* d_softmax_probs = nullptr;       // [1000] (ResNet)

// ============================================================================
// Kernel 1: 置信度过滤 + 坐标解码 (YOLO)
// ============================================================================
__global__ void filter_decode_yolo_kernel(
    const float* __restrict__ d_output,    // [attr_size, num_dets]
    int num_detections,
    int num_classes,
    float conf_thres,
    float scale,
    float dw,
    float dh,
    float* __restrict__ d_filtered_boxes,  // [MAX_DETS, 4]
    int* __restrict__ d_filtered_class_ids,// [MAX_DETS]
    float* __restrict__ d_filtered_conf,   // [MAX_DETS]
    int* __restrict__ d_filtered_indices,  // [MAX_DETS] - 原始索引
    int* __restrict__ d_filtered_count     // [1]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_detections) return;

    // 内存布局: [attr_size, num_detections]
    // cx[0], cy[1], w[2], h[3], classes[4:4+num_classes], ...

    // 读取 bbox
    float cx = d_output[0 * num_detections + idx];
    float cy = d_output[1 * num_detections + idx];
    float w = d_output[2 * num_detections + idx];
    float h = d_output[3 * num_detections + idx];

    // 找最大类别分数
    float max_score = -1.0f;
    int max_class_id = -1;

    for (int c = 0; c < num_classes; ++c) {
        float score = d_output[(4 + c) * num_detections + idx];
        if (score > max_score) {
            max_score = score;
            max_class_id = c;
        }
    }

    // 置信度过滤
    if (max_score > conf_thres) {
        // 原子计数获取写入位置
        int write_idx = atomicAdd(d_filtered_count, 1);

        // 坐标解码: letterbox -> 原图
        float x = (cx - 0.5f * w - dw) / scale;
        float y = (cy - 0.5f * h - dh) / scale;
        float width = w / scale;
        float height = h / scale;

        // 写入结果
        d_filtered_boxes[write_idx * 4 + 0] = x;
        d_filtered_boxes[write_idx * 4 + 1] = y;
        d_filtered_boxes[write_idx * 4 + 2] = width;
        d_filtered_boxes[write_idx * 4 + 3] = height;
        d_filtered_class_ids[write_idx] = max_class_id;
        d_filtered_conf[write_idx] = max_score;
        d_filtered_indices[write_idx] = idx;  // 保存原始索引
    }
}

// ============================================================================
// Kernel 2: 计算 IoU (用于 NMS)
// ============================================================================
__device__ float calculate_iou(const float* a, const float* b) {
    float x1 = fmaxf(a[0], b[0]);
    float y1 = fmaxf(a[1], b[1]);
    float x2 = fminf(a[0] + a[2], b[0] + b[2]);
    float y2 = fminf(a[1] + a[3], b[1] + b[3]);

    float inter_area = fmaxf(0.0f, x2 - x1) * fmaxf(0.0f, y2 - y1);
    float a_area = a[2] * a[3];
    float b_area = b[2] * b[3];
    float union_area = a_area + b_area - inter_area;

    return (union_area > 0.0f) ? (inter_area / union_area) : 0.0f;
}

__global__ void compute_iou_matrix_kernel(
    const float* __restrict__ d_boxes,     // [N, 4]
    int num_boxes,
    float* __restrict__ d_iou_matrix       // [N, N]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= num_boxes || idy >= num_boxes) return;

    float iou = (idx == idy) ? 1.0f : calculate_iou(
        &d_boxes[idx * 4],
        &d_boxes[idy * 4]
    );

    d_iou_matrix[idx * num_boxes + idy] = iou;
}

// ============================================================================
// Kernel 3: NMS (并行处理 - 修复版)
// ============================================================================
__global__ void nms_suppress_kernel(
    const float* __restrict__ d_iou_matrix,
    const float* __restrict__ d_conf,
    int num_boxes,
    float nms_thres,
    int* __restrict__ d_suppressed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;

    // 如果已经被抑制，跳过
    if (d_suppressed[idx]) return;

    // 抑制 IoU 超过阈值的框（仅抑制置信度更低的）
    // 为了避免并发问题，每个线程只抑制置信度严格更低的框
    for (int j = 0; j < num_boxes; ++j) {
        if (idx == j) continue;
        if (d_suppressed[j]) continue;

        float iou = d_iou_matrix[idx * num_boxes + j];
        if (iou > nms_thres && d_conf[idx] > d_conf[j]) {
            // 使用原子操作确保只设置一次
            int old = 0;
            if (atomicCAS(&d_suppressed[j], 0, 1) == 0) {
                // 成功抑制
            }
        }
    }
}

// ============================================================================
// Kernel 4: 收集未抑制的索引
// ============================================================================
__global__ void collect_keep_indices_kernel(
    const int* __restrict__ d_suppressed,
    int num_boxes,
    int* __restrict__ d_keep_indices,
    int* __restrict__ d_keep_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;

    if (!d_suppressed[idx]) {
        int write_idx = atomicAdd(d_keep_count, 1);
        d_keep_indices[write_idx] = idx;
    }
}

// ============================================================================
// Kernel 5: ResNet Softmax
// ============================================================================
__global__ void softmax_find_max_kernel(
    const float* __restrict__ d_logits,
    int num_classes,
    float* __restrict__ d_max_val,  // [1] - 需要后续 reduce
    int* __restrict__ d_max_idx     // [1] - 需要后续 reduce
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_classes) return;

    // 简单实现：每个线程保存局部最大值
    // 实际应用中可以使用 cub::DeviceReduce
    extern __shared__ float s_max_val[];
    __shared__ int s_max_idx;
    __shared__ bool s_first;

    if (threadIdx.x == 0) {
        s_first = true;
        s_max_idx = -1;
    }
    __syncthreads();

    // 原子更新最大值
    float val = d_logits[idx];
    if (s_first) {
        s_max_val[0] = val;
        s_max_idx = idx;
        s_first = false;
    } else {
        if (val > s_max_val[0]) {
            s_max_val[0] = val;
            s_max_idx = idx;
        }
    }
    __syncthreads();

    // 第一个线程写入全局内存
    if (threadIdx.x == 0) {
        d_max_val[0] = s_max_val[0];
        d_max_idx[0] = s_max_idx;
    }
}

__global__ void softmax_compute_kernel(
    const float* __restrict__ d_logits,
    int num_classes,
    float max_val,
    float* __restrict__ d_probs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_classes) return;

    // exp(logit - max) for numerical stability
    d_probs[idx] = expf(d_logits[idx] - max_val);
}

__global__ void softmax_sum_kernel(
    const float* __restrict__ d_probs,
    int num_classes,
    float* __restrict__ d_sum  // [1]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_classes) return;

    // 并行求和
    extern __shared__ float s_sum[];
    s_sum[threadIdx.x] = d_probs[idx];
    __syncthreads();

    // 块内归约
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(d_sum, s_sum[0]);
    }
}

// ============================================================================
// Kernel 6: 分割 mask 系数收集
// ============================================================================
__global__ void collect_mask_coeffs_kernel(
    const float* __restrict__ d_bbox_output,
    const int* __restrict__ d_keep_indices,
    int num_keep,
    int num_detections,
    int num_classes,
    int mask_coeff_len,
    float* __restrict__ d_mask_coeffs  // [num_keep, mask_coeff_len]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_keep) return;

    int det_idx = d_keep_indices[idx];
    int mask_offset = 4 + num_classes;  // mask 系数起始位置

    for (int c = 0; c < mask_coeff_len; ++c) {
        d_mask_coeffs[idx * mask_coeff_len + c] =
            d_bbox_output[(mask_offset + c) * num_detections + det_idx];
    }
}

// ============================================================================
// Host 函数：初始化
// ============================================================================
void cuda_postprocess_init() {
    // YOLO 后处理缓冲区
    cudaMalloc(&d_filtered_boxes, MAX_DETECTIONS * 4 * sizeof(float));
    cudaMalloc(&d_filtered_class_ids, MAX_DETECTIONS * sizeof(int));
    cudaMalloc(&d_filtered_conf, MAX_DETECTIONS * sizeof(float));
    cudaMalloc(&d_filtered_indices, MAX_DETECTIONS * sizeof(int));  // 原始索引
    cudaMalloc(&d_filtered_count, sizeof(int));
    cudaMalloc(&d_iou_matrix, MAX_DETECTIONS * MAX_DETECTIONS * sizeof(float));
    cudaMalloc(&d_suppressed, MAX_DETECTIONS * sizeof(int));
    cudaMalloc(&d_keep_indices, MAX_DETECTIONS * sizeof(int));
    cudaMalloc(&d_keep_count, sizeof(int));

    // ResNet Softmax 缓冲区
    cudaMalloc(&d_softmax_probs, 1000 * sizeof(float));

    CUDA_CHECK(cudaGetLastError());
}

void cuda_postprocess_destroy() {
    cudaFree(d_filtered_boxes);
    cudaFree(d_filtered_class_ids);
    cudaFree(d_filtered_conf);
    cudaFree(d_filtered_indices);  // 原始索引
    cudaFree(d_filtered_count);
    cudaFree(d_iou_matrix);
    cudaFree(d_suppressed);
    cudaFree(d_keep_indices);
    cudaFree(d_keep_count);
    cudaFree(d_softmax_probs);
}

// ============================================================================
// Host 函数：YOLO 检测后处理
// ============================================================================
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
) {
    // 1. 初始化计数器
    CUDA_CHECK(cudaMemsetAsync(d_filtered_count, 0, sizeof(int), stream));

    // 2. 计算 letterbox 参数
    float scale = fminf((float)input_h / img_h, (float)input_w / img_w);
    float dw = (input_w - scale * img_w) / 2.0f;
    float dh = (input_h - scale * img_h) / 2.0f;

    // 3. GPU 置信度过滤 + 坐标解码
    int threads = THREADS_PER_BLOCK;
    int blocks = (num_detections + threads - 1) / threads;

    filter_decode_yolo_kernel<<<blocks, threads, 0, stream>>>(
        d_output, num_detections, num_classes, conf_thres,
        scale, dw, dh,
        d_filtered_boxes, d_filtered_class_ids, d_filtered_conf, d_filtered_indices, d_filtered_count
    );
    CUDA_CHECK(cudaGetLastError());

    // 4. 拷贝过滤后的结果到 Host
    int filtered_count;
    cudaMemcpyAsync(&filtered_count, d_filtered_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (filtered_count == 0) {
        output.clear();
        return;
    }

    // 拷贝所有过滤后的结果
    std::vector<float> h_boxes(filtered_count * 4);
    std::vector<int> h_class_ids(filtered_count);
    std::vector<float> h_conf(filtered_count);

    cudaMemcpyAsync(h_boxes.data(), d_filtered_boxes, filtered_count * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_class_ids.data(), d_filtered_class_ids, filtered_count * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_conf.data(), d_filtered_conf, filtered_count * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 5. CPU NMS (使用 OpenCV)
    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    std::vector<float> confidences;

    boxes.reserve(filtered_count);
    class_ids.reserve(filtered_count);
    confidences.reserve(filtered_count);

    for (int i = 0; i < filtered_count; ++i) {
        boxes.push_back(cv::Rect(
            static_cast<int>(h_boxes[i * 4 + 0]),
            static_cast<int>(h_boxes[i * 4 + 1]),
            static_cast<int>(h_boxes[i * 4 + 2]),
            static_cast<int>(h_boxes[i * 4 + 3])
        ));
        class_ids.push_back(h_class_ids[i]);
        confidences.push_back(h_conf[i]);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.0f, nms_thres, indices);

    // 6. 组装最终输出
    output.clear();
    output.reserve(indices.size());

    for (int idx : indices) {
        Detection det;
        det.class_id = class_ids[idx];
        det.conf = confidences[idx];
        det.bbox = boxes[idx];
        output.push_back(det);
    }
}

// ============================================================================
// Host 函数：ResNet 分类后处理
// ============================================================================
void cuda_postprocess_resnet(
    float* d_output,
    int num_classes,
    int& cls_id,
    float& cls_score,
    cudaStream_t stream
) {
    // Debug: 打印原始 logits
    // std::vector<float> h_logits(num_classes);
    // cudaMemcpyAsync(h_logits.data(), d_output, num_classes * sizeof(float), cudaMemcpyDeviceToHost, stream);
    // cudaStreamSynchronize(stream);
    // printf("  [DEBUG] TensorRT Raw Logits: [");
    // for (int i = 0; i < num_classes; ++i) {
    //     printf("%.4f%s", h_logits[i], (i < num_classes - 1) ? ", " : "");
    // }
    // printf("]\n");

    // 1. 找最大值
    float* d_max_val;
    int* d_max_idx;
    cudaMalloc(&d_max_val, sizeof(float));
    cudaMalloc(&d_max_idx, sizeof(int));

    int threads = THREADS_PER_BLOCK;
    int blocks = (num_classes + threads - 1) / threads;

    softmax_find_max_kernel<<<blocks, threads, sizeof(float), stream>>>(
        d_output, num_classes, d_max_val, d_max_idx
    );
    CUDA_CHECK(cudaGetLastError());

    // 2. 计算 exp(logit - max)
    float h_max_val;
    cudaMemcpyAsync(&h_max_val, d_max_val, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    softmax_compute_kernel<<<blocks, threads, 0, stream>>>(
        d_output, num_classes, h_max_val, d_softmax_probs
    );
    CUDA_CHECK(cudaGetLastError());

    // 3. 计算概率和
    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    CUDA_CHECK(cudaMemsetAsync(d_sum, 0, sizeof(float), stream));

    softmax_sum_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(
        d_softmax_probs, num_classes, d_sum
    );
    CUDA_CHECK(cudaGetLastError());

    float h_sum;
    cudaMemcpyAsync(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 4. 找最大概率
    std::vector<float> h_probs(num_classes);
    cudaMemcpyAsync(h_probs.data(), d_softmax_probs, num_classes * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    float best_prob = 0.0f;
    int best_idx = 0;
    for (int i = 0; i < num_classes; ++i) {
        float p = h_probs[i] / h_sum;
        if (p > best_prob) {
            best_prob = p;
            best_idx = i;
        }
    }

    cls_id = best_idx;
    cls_score = best_prob;

    cudaFree(d_max_val);
    cudaFree(d_max_idx);
    cudaFree(d_sum);
}

// ============================================================================
// Host 函数：YOLO 分割后处理
// ============================================================================
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
) {
    // 1. GPU 置信度过滤 + 坐标解码
    CUDA_CHECK(cudaMemsetAsync(d_filtered_count, 0, sizeof(int), stream));

    float scale = fminf((float)input_h / img_h, (float)input_w / img_w);
    float dw = (input_w - scale * img_w) / 2.0f;
    float dh = (input_h - scale * img_h) / 2.0f;

    int threads = THREADS_PER_BLOCK;
    int blocks = (num_detections + threads - 1) / threads;

    filter_decode_yolo_kernel<<<blocks, threads, 0, stream>>>(
        d_bbox_output, num_detections, num_classes, conf_thres,
        scale, dw, dh,
        d_filtered_boxes, d_filtered_class_ids, d_filtered_conf, d_filtered_indices, d_filtered_count
    );
    CUDA_CHECK(cudaGetLastError());

    // 2. 拷贝过滤后的结果到 Host
    int filtered_count;
    cudaMemcpyAsync(&filtered_count, d_filtered_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (filtered_count == 0) {
        output.clear();
        return;
    }

    // 拷贝所有过滤后的结果
    std::vector<float> h_boxes(filtered_count * 4);
    std::vector<int> h_class_ids(filtered_count);
    std::vector<float> h_conf(filtered_count);
    std::vector<int> h_indices(filtered_count);  // 原始索引

    cudaMemcpyAsync(h_boxes.data(), d_filtered_boxes, filtered_count * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_class_ids.data(), d_filtered_class_ids, filtered_count * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_conf.data(), d_filtered_conf, filtered_count * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_indices.data(), d_filtered_indices, filtered_count * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 3. CPU NMS
    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    std::vector<float> confidences;

    boxes.reserve(filtered_count);
    class_ids.reserve(filtered_count);
    confidences.reserve(filtered_count);

    for (int i = 0; i < filtered_count; ++i) {
        boxes.push_back(cv::Rect(
            static_cast<int>(h_boxes[i * 4 + 0]),
            static_cast<int>(h_boxes[i * 4 + 1]),
            static_cast<int>(h_boxes[i * 4 + 2]),
            static_cast<int>(h_boxes[i * 4 + 3])
        ));
        class_ids.push_back(h_class_ids[i]);
        confidences.push_back(h_conf[i]);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.0f, nms_thres, indices);

    if (indices.empty()) {
        output.clear();
        return;
    }

    // 4. 收集 mask 系数
    int num_keep = indices.size();

    // 拷贝原始 bbox 输出用于提取 mask 系数
    std::vector<float> h_bbox_output(detection_attr_size * num_detections);
    cudaMemcpyAsync(h_bbox_output.data(), d_bbox_output,
                    detection_attr_size * num_detections * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 收集 mask 系数
    std::vector<float> mask_coeffs;
    mask_coeffs.reserve(num_keep * mask_coeff_len);

    for (int i = 0; i < num_keep; ++i) {
        // indices[i] 是 boxes 中的索引，h_indices[indices[i]] 是原始 GPU 输出中的索引
        int filtered_idx = indices[i];
        int original_idx = h_indices[filtered_idx];

        int mask_offset = 4 + num_classes;
        for (int c = 0; c < mask_coeff_len; ++c) {
            mask_coeffs.push_back(h_bbox_output[(mask_offset + c) * num_detections + original_idx]);
        }
    }

    // 5. GPU mask 解码
    float* d_mask_coeffs;
    unsigned char* d_mask_out;
    int mask_out_h = mask_proto_h;
    int mask_out_w = mask_proto_w;

    cudaMalloc(&d_mask_coeffs, num_keep * mask_coeff_len * sizeof(float));
    cudaMalloc(&d_mask_out, num_keep * mask_out_h * mask_out_w * sizeof(unsigned char));

    cudaMemcpyAsync(d_mask_coeffs, mask_coeffs.data(), num_keep * mask_coeff_len * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // 调用 preprocess.cu 中的 mask 解码函数
    cuda_decode_masks(
        d_mask_coeffs, d_mask_proto, d_mask_out,
        num_keep, mask_coeff_len, mask_proto_h, mask_proto_w,
        mask_out_h, mask_out_w, stream
    );
    cudaStreamSynchronize(stream);

    // 6. 拷贝 mask 结果到 Host
    std::vector<unsigned char> h_mask_data(num_keep * mask_out_h * mask_out_w);
    cudaMemcpyAsync(h_mask_data.data(), d_mask_out, h_mask_data.size(),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 7. 组装结果（裁剪 mask 到 bbox 尺寸）
    output.clear();
    output.reserve(num_keep);

    for (int i = 0; i < num_keep; ++i) {
        int idx = indices[i];
        DetectionWithMask det;
        det.class_id = class_ids[idx];
        det.conf = confidences[idx];
        det.bbox = boxes[idx];

        // 计算在 mask 原型空间中的 bbox 位置
        // boxes[idx] 是 NMS 后的原图坐标，需要映射回 letterbox 空间
        // 反向转换：原图 * scale + padding = letterbox
        float letterbox_x = boxes[idx].x * scale + dw;
        float letterbox_y = boxes[idx].y * scale + dh;
        float letterbox_w = boxes[idx].width * scale;
        float letterbox_h = boxes[idx].height * scale;

        // bbox 中心点在 letterbox 空间
        float cx = letterbox_x + 0.5f * letterbox_w;
        float cy = letterbox_y + 0.5f * letterbox_h;

        // 映射到 mask 原型空间 (下采样 4 倍)
        // 注意：应该用 bbox 左上角而不是中心点作为起点
        float mask_x = letterbox_x / 4.0f;
        float mask_y = letterbox_y / 4.0f;
        float mask_w = letterbox_w / 4.0f;
        float mask_h = letterbox_h / 4.0f;

        // 计算裁剪区域
        int mask_x0 = std::max(0, static_cast<int>(std::round(mask_x)));
        int mask_y0 = std::max(0, static_cast<int>(std::round(mask_y)));
        int mask_x1 = std::min(mask_out_w, static_cast<int>(std::round(mask_x + mask_w)) + 1);
        int mask_y1 = std::min(mask_out_h, static_cast<int>(std::round(mask_y + mask_h)) + 1);

        int crop_w = mask_x1 - mask_x0;
        int crop_h = mask_y1 - mask_y0;

        if (crop_w > 0 && crop_h > 0) {
            auto mask = std::make_shared<InstanceSegmentMap>(crop_w, crop_h);

            const unsigned char* src = h_mask_data.data() + i * mask_out_h * mask_out_w;

            for (int y = 0; y < crop_h; ++y) {
                for (int x = 0; x < crop_w; ++x) {
                    int src_y = mask_y0 + y;
                    int src_x = mask_x0 + x;
                    if (src_y < mask_out_h && src_x < mask_out_w) {
                        mask->data[y * crop_w + x] = src[src_y * mask_out_w + src_x];
                    }
                }
            }

            det.mask = mask;
        }

        output.push_back(det);
    }

    cudaFree(d_mask_coeffs);
    cudaFree(d_mask_out);
}


