#pragma once
#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

enum class ModelType {
    YOLO_DETECT,    // 支持 v8, v11
    YOLO_SEG,       // 支持 v8-seg, v11-seg
    RESNET_CLS      // 支持 ResNet, Cls
};

// 基础检测结果结构体
struct Detection {
    int class_id;
    float conf;
    cv::Rect bbox;
};

// 实例分割 Mask 存储
struct InstanceSegmentMap {
    int width;
    int height;
    std::vector<uint8_t> data;  // 二值 mask (0 或 255)

    InstanceSegmentMap(int w = 0, int h = 0) : width(w), height(h) {
        if (w > 0 && h > 0) data.resize(w * h);
    }
};

// 带分割的检测结果
struct DetectionWithMask {
    int class_id;
    float conf;
    cv::Rect bbox;
    std::shared_ptr<InstanceSegmentMap> mask;  // 可选的 mask

    DetectionWithMask() : class_id(-1), conf(0.0f), bbox(0,0,0,0), mask(nullptr) {}
};

struct ModelConfig {
    ModelType type;
    std::string engine_file;

    // 输入维度
    int input_w;
    int input_h;

    // 归一化参数 (ResNet需要)
    float mean[3] = {0.0f, 0.0f, 0.0f};
    float std[3]  = {1.0f, 1.0f, 1.0f};

    // YOLO 特有
    float conf_thres = 0.25f;
    float nms_thres = 0.45f;
    int num_classes = 80; // YOLOv8/11 coco默认80, ResNet ImageNet默认1000

    // 分割特有
    int mask_proto_size = 160; // mask 原型尺寸 (通常 160x160)
};