#pragma once
#include <string>
#include <vector>

enum class ModelType {
    YOLO_DETECTION, // 支持 v8, v11
    RESNET_CLS      // 支持 ResNet, Cls
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
};