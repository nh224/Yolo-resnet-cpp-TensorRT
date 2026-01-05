#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"

struct Object {
    cv::Rect rect;
    int label;
    float prob;
};

enum class ModelType {
    YOLO_DETECT, 
    RESNET_CLS   
};

struct RunConfig {
    ModelType type;
    std::string engine_file_path;
    float conf_thres = 0.25f;
    float nms_thres = 0.45f;
    int num_classes = 80; 
};

class Wrapper {
public:
    Wrapper(const RunConfig& config);
    ~Wrapper();

    void infer(const cv::Mat& input, std::vector<Object>& objects, int& cls_id, float& cls_score);

private:
    void init_context();
    void postprocess_yolo(float* output, int output_size, std::vector<Object>& objects, float scale, int img_w, int img_h);
    void postprocess_resnet(float* output, int output_size, int& cls_id, float& cls_score);

private:
    RunConfig config_;
    
    // TensorRT components
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t stream = nullptr;

    // Memory buffers
    void* buffers[2]; // 0: Input, 1: Output
    
    // TensorRT 10 需要保存输入输出的名字
    std::string input_name;
    std::string output_name;

    float* output_buffer_host = nullptr;
    
    // Model Dimensions
    int input_w = 0;
    int input_h = 0;
    int input_size = 0;
    int output_size = 0;
};