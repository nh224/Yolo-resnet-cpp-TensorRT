#include "models/segmentation_model.h"
#include "preprocess.h"
#include "postprocess.h"
#include <chrono>

SegmentationModel::SegmentationModel(const SegmentationConfig& config)
    : ModelBase(config.engine_file, true), config_(config) {
    // 获取 mask 维度
    get_mask_dims(mask_coeff_len, mask_proto_h, mask_proto_w);
}

SegmentationModel::~SegmentationModel() {
    // 基类析构函数会自动清理资源
}

void SegmentationModel::infer(const cv::Mat& input) {
    using namespace std::chrono;

    float mean[3] = {0.0f, 0.0f, 0.0f};
    float std_val[3] = {1.0f, 1.0f, 1.0f};
    PreprocessMode p_mode = MODE_LETTERBOX;

    // =================================================================
    // 1. 预处理
    // =================================================================
    auto t0 = high_resolution_clock::now();

    cuda_preprocess((uint8_t*)input.data, input.cols, input.rows,
                    (float*)gpu_buffers[0], input_w, input_h, stream,
                    mean, std_val, p_mode);
    cudaStreamSynchronize(stream);

    auto t1 = high_resolution_clock::now();

    // =================================================================
    // 2. 推理
    // =================================================================
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    auto t2 = high_resolution_clock::now();

    // =================================================================
    // 3. 后处理
    // =================================================================
    // CUDA 后处理：bbox 在 gpu_buffers[1]，mask_proto 在 gpu_buffers[2]
    cuda_postprocess_yolo_seg(
        (float*)gpu_buffers[1], (float*)gpu_buffers[2],
        detection_attribute_size, num_detections,
        config_.num_classes, mask_coeff_len, mask_proto_h, mask_proto_w,
        config_.conf_thres,
        input.cols, input.rows, input_w, input_h,
        cached_results_, config_.nms_thres, stream
    );

    auto t3 = high_resolution_clock::now();

    // 打印耗时
    double ms_pre   = duration<double, std::milli>(t1 - t0).count();
    double ms_infer = duration<double, std::milli>(t2 - t1).count();
    double ms_post  = duration<double, std::milli>(t3 - t2).count();

    printf("Time: Pre %.2f ms | Infer %.2f ms | Post %.2f ms | Total %.2f ms\n",
           ms_pre, ms_infer, ms_post, ms_pre + ms_infer + ms_post);
}

void SegmentationModel::get_results(std::vector<DetectionWithMask>& objects) {
    objects = cached_results_;
}
