#include "models/detection_model.h"
#include "preprocess.h"
#include "postprocess.h"
#include <chrono>

DetectionModel::DetectionModel(const DetectionConfig& config)
    : ModelBase(config.engine_file, false), config_(config) {
}

DetectionModel::~DetectionModel() {
    // 基类析构函数会自动清理资源
}

void DetectionModel::infer(const cv::Mat& input) {
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
    // 3. 后处理 (GPU 加速)
    // =================================================================
    // 结果暂存在成员变量中，通过 get_results() 获取
    cuda_postprocess_yolo(
        (float*)gpu_buffers[1], detection_attribute_size, num_detections,
        config_.num_classes, config_.conf_thres,
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

void DetectionModel::get_results(std::vector<Detection>& objects) {
    objects = cached_results_;
}
