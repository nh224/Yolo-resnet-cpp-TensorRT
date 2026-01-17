#pragma once
#include "core/model_base.h"
#include "types.h"
#include <vector>

// YOLO 分割模型配置
struct SegmentationConfig {
    std::string engine_file;
    int num_classes = 80;
    float conf_thres = 0.25f;
    float nms_thres = 0.45f;
};

// YOLO 分割模型
class SegmentationModel : public ModelBase {
public:
    SegmentationModel(const SegmentationConfig& config);
    ~SegmentationModel();

    // 执行推理
    void infer(const cv::Mat& input) override;

    // 获取检测结果
    void get_results(std::vector<DetectionWithMask>& objects);

private:
    SegmentationConfig config_;
    std::vector<DetectionWithMask> cached_results_;  // 缓存推理结果

    // 分割特有维度
    int mask_coeff_len = 0;
    int mask_proto_h = 0;
    int mask_proto_w = 0;
};
