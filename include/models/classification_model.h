#pragma once
#include "core/model_base.h"
#include "types.h"

// 分类模型配置
struct ClassificationConfig {
    std::string engine_file;
    int num_classes = 1000;
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3] = {0.229f, 0.224f, 0.225f};
    bool use_normalization = true;  // 是否使用归一化（false 则直接用 [0,1]）
};

// ResNet 等分类模型
class ClassificationModel : public ModelBase {
public:
    ClassificationModel(const ClassificationConfig& config);
    ~ClassificationModel();

    // 执行推理
    void infer(const cv::Mat& input) override;

    // 获取分类结果
    void get_results(int& class_id, float& score);

private:
    ClassificationConfig config_;
    int cached_class_id_;
    float cached_score_;
};
