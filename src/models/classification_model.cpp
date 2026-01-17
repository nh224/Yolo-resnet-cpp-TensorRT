#include "models/classification_model.h"
#include "preprocess.h"
#include "postprocess.h"
#include <chrono>

ClassificationModel::ClassificationModel(const ClassificationConfig& config)
    : ModelBase(config.engine_file, false), config_(config),
      cached_class_id_(-1), cached_score_(0.0f) {
}

ClassificationModel::~ClassificationModel() {
    // 基类析构函数会自动清理资源
}

void ClassificationModel::infer(const cv::Mat& input) {
    using namespace std::chrono;

    float mean[3], std_val[3];
    PreprocessMode p_mode;

    // 使用配置中的归一化参数
    if (config_.use_normalization) {
        mean[0] = config_.mean[0];
        mean[1] = config_.mean[1];
        mean[2] = config_.mean[2];
        std_val[0] = config_.std[0];
        std_val[1] = config_.std[1];
        std_val[2] = config_.std[2];
    } else {
        // 不使用归一化，直接用 [0,1]
        mean[0] = 0.0f; mean[1] = 0.0f; mean[2] = 0.0f;
        std_val[0] = 1.0f; std_val[1] = 1.0f; std_val[2] = 1.0f;
    }
    // 使用 CenterCrop 模式以匹配 PyTorch 的标准 ImageNet 预处理
    p_mode = MODE_CENTER_CROP;

    // =================================================================
    // 1. 预处理
    // =================================================================
    auto t0 = high_resolution_clock::now();

    // Debug: 打印原始图像信息
    // printf("  [DEBUG] Input image: %dx%d, channels=%d, elemSize=%zu\n",
    //        input.cols, input.rows, input.channels(), input.elemSize());

    cuda_preprocess((uint8_t*)input.data, input.cols, input.rows,
                    (float*)gpu_buffers[0], input_w, input_h, stream,
                    mean, std_val, p_mode);
    cudaStreamSynchronize(stream);

    auto t1 = high_resolution_clock::now();

    // =================================================================
    // 2. 推理
    // =================================================================

    // Debug: 打印预处理后的前10个值
    // std::vector<float> h_input_debug(15);
    // cudaMemcpyAsync(h_input_debug.data(), (float*)gpu_buffers[0], 15 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    // cudaStreamSynchronize(stream);
    // printf("  [DEBUG] Preprocessed input (first 15 values): [");
    // for (int i = 0; i < 15; ++i) {
    //     printf("%.4f%s", h_input_debug[i], (i < 14) ? ", " : "");
    // }
    // printf("]\n");

    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    auto t2 = high_resolution_clock::now();

    // =================================================================
    // 3. 后处理
    // =================================================================
    cuda_postprocess_resnet(
        (float*)gpu_buffers[1], config_.num_classes,
        cached_class_id_, cached_score_, stream
    );

    auto t3 = high_resolution_clock::now();

    // 打印耗时
    double ms_pre   = duration<double, std::milli>(t1 - t0).count();
    double ms_infer = duration<double, std::milli>(t2 - t1).count();
    double ms_post  = duration<double, std::milli>(t3 - t2).count();

    printf("Time: Pre %.2f ms | Infer %.2f ms | Post %.2f ms | Total %.2f ms\n",
           ms_pre, ms_infer, ms_post, ms_pre + ms_infer + ms_post);
}

void ClassificationModel::get_results(int& class_id, float& score) {
    class_id = cached_class_id_;
    score = cached_score_;
}
