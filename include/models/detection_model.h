#pragma once
#include "core/model_base.h"
#include "types.h"
#include <vector>

// YOLO 检测模型配置
struct DetectionConfig {
    std::string engine_file;
    int num_classes = 80;
    float conf_thres = 0.25f;
    float nms_thres = 0.45f;
};

// YOLO 检测模型
class DetectionModel : public ModelBase {
public:
    DetectionModel(const DetectionConfig& config);
    ~DetectionModel();

    // 执行推理
    void infer(const cv::Mat& input) override;

    // 获取检测结果
    void get_results(std::vector<Detection>& objects);

private:
    DetectionConfig config_;
    std::vector<Detection> cached_results_;  // 缓存推理结果
};
