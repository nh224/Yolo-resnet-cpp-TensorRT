#include "wrapper.h"
#include "preprocess.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath> // for exp

// 简单的 Logger
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

Wrapper::Wrapper(const RunConfig& config) : config_(config) {
    init_context();
    cuda_preprocess_init(4096 * 4096); 
}

Wrapper::~Wrapper() {
    if (stream) cudaStreamDestroy(stream);
    if (buffers[0]) cudaFree(buffers[0]);
    if (buffers[1]) cudaFree(buffers[1]);
    if (output_buffer_host) delete[] output_buffer_host;
    
    if (context) delete context;
    if (engine) delete engine;
    if (runtime) delete runtime;

    cuda_preprocess_destroy();
}

void Wrapper::init_context() {
    // 1. 读取 Engine 文件
    std::ifstream file(config_.engine_file_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error: Could not read engine file: " << config_.engine_file_path << std::endl;
        exit(-1);
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    char* trtModelStream = new char[size];
    file.read(trtModelStream, size);
    file.close();

    // 2. 创建 Runtime
    runtime = nvinfer1::createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    context = engine->createExecutionContext();
    delete[] trtModelStream;

    cudaStreamCreate(&stream);

    // 3. 解析 TensorRT 10 输入输出信息 (添加了调试打印)
    int nbIOTensors = engine->getNbIOTensors();
    std::cout << "\n=== Model I/O Info ===" << std::endl;
    
    for (int i = 0; i < nbIOTensors; ++i) {
        const char* name = engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = engine->getTensorIOMode(name);
        nvinfer1::Dims dims = engine->getTensorShape(name);

        std::cout << "Tensor " << i << ": Name=" << name 
                  << " Mode=" << (mode == nvinfer1::TensorIOMode::kINPUT ? "Input" : "Output")
                  << " Dims=[";
        
        int vol = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            std::cout << dims.d[j] << (j < dims.nbDims - 1 ? "," : "");
            // 计算大小时忽略 -1 (动态维度)
            if(dims.d[j] > 0) vol *= dims.d[j];
        }
        std::cout << "]" << std::endl;

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_name = name;
            input_size = vol;
            
            // 维度解析逻辑
            if(dims.nbDims == 4) {
                input_h = dims.d[2];
                input_w = dims.d[3];
            } else if (dims.nbDims == 3) {
                input_h = dims.d[1];
                input_w = dims.d[2];
            }
            
            // 容错：如果是动态输入 [-1, 3, -1, -1]，我们根据模型类型强制设定默认值
            if (input_h <= 0 || input_w <= 0) {
                if (config_.type == ModelType::RESNET_CLS) {
                    input_h = 224; input_w = 224;
                } else {
                    input_h = 640; input_w = 640;
                }
                std::cout << "Warning: Dynamic shape detected. Forcing input to " << input_w << "x" << input_h << std::endl;
            }

            cudaMalloc(&buffers[0], input_size * sizeof(float));
        } 
        else {
            output_name = name;
            output_size = vol;
            cudaMalloc(&buffers[1], output_size * sizeof(float));
        }
    }
    std::cout << "======================\n" << std::endl;
    
    output_buffer_host = new float[output_size];
}

void Wrapper::infer(const cv::Mat& input, std::vector<Object>& objects, int& cls_id, float& cls_score) {
    // 1. 预处理参数
    float mean[3], std[3];
    PreprocessMode p_mode;

    if (config_.type == ModelType::RESNET_CLS) {
        mean[0] = 0.485f; mean[1] = 0.456f; mean[2] = 0.406f;
        std[0]  = 0.229f; std[1]  = 0.224f; std[2]  = 0.225f;
        p_mode = MODE_STRETCH;
    } else {
        mean[0] = 0.0f; mean[1] = 0.0f; mean[2] = 0.0f;
        std[0]  = 1.0f; std[1]  = 1.0f; std[2]  = 1.0f;
        p_mode = MODE_LETTERBOX;
    }

    // 2. 执行预处理
    cuda_preprocess(
        input.data, input.cols, input.rows, 
        (float*)buffers[0], input_w, input_h, 
        stream, mean, std, p_mode
    );

    // ==========================================================
    // TensorRT 10.x 关键修复: 必须设置 Input Shape !
    // ==========================================================
    nvinfer1::Dims input_dims;
    input_dims.nbDims = 4;
    input_dims.d[0] = 1;        // Batch Size 固定为 1
    input_dims.d[1] = 3;        // Channels
    input_dims.d[2] = input_h;  // Height
    input_dims.d[3] = input_w;  // Width
    
    // 显式设置当前推理的维度 (解决动态 Batch 导致的推理失败)
    context->setInputShape(input_name.c_str(), input_dims);

    context->setInputTensorAddress(input_name.c_str(), buffers[0]);
    context->setTensorAddress(output_name.c_str(), buffers[1]); 

    // 执行推理
    bool status = context->enqueueV3(stream);
    if (!status) {
        std::cerr << "Error: TensorRT enqueueV3 failed!" << std::endl;
    }

    // 拷贝结果
    cudaMemcpyAsync(output_buffer_host, buffers[1], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 3. 后处理
    if (config_.type == ModelType::YOLO_DETECT) {
        float scale = std::min((float)input_h / input.rows, (float)input_w / input.cols);
        postprocess_yolo(output_buffer_host, output_size, objects, scale, input.cols, input.rows);
    } 
    else {
        postprocess_resnet(output_buffer_host, output_size, cls_id, cls_score);
    }
}

// ---------------- ResNet Postprocess (含 Softmax) ----------------
void Wrapper::postprocess_resnet(float* output, int output_size, int& cls_id, float& cls_score) {
    // 1. 实现 Softmax 将 Logits 转为概率
    std::vector<float> probs(output_size);
    float max_logit = -1e9;
    
    // 找最大值防止溢出
    for(int i=0; i<output_size; ++i) {
        if(output[i] > max_logit) max_logit = output[i];
    }
    
    float sum = 0.0f;
    for(int i=0; i<output_size; ++i) {
        probs[i] = std::exp(output[i] - max_logit);
        sum += probs[i];
    }
    
    // 2. 找最大概率
    int best_idx = 0;
    float best_prob = 0.0f;
    
    for(int i=0; i<output_size; ++i) {
        probs[i] /= sum; // 归一化
        if(probs[i] > best_prob) {
            best_prob = probs[i];
            best_idx = i;
        }
    }
    
    cls_id = best_idx;
    cls_score = best_prob;
}

// ---------------- YOLO Postprocess ----------------
void Wrapper::postprocess_yolo(float* output, int output_size, std::vector<Object>& objects, float scale, int img_w, int img_h) {
    int num_classes = config_.num_classes;
    int num_anchors = 8400; 
    
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < num_anchors; ++i) {
        float max_cls_score = 0.0f;
        int max_cls_id = -1;

        for (int c = 0; c < num_classes; ++c) {
            float score = output[(4 + c) * num_anchors + i];
            if (score > max_cls_score) {
                max_cls_score = score;
                max_cls_id = c;
            }
        }

        if (max_cls_score > config_.conf_thres) {
            float cx = output[0 * num_anchors + i];
            float cy = output[1 * num_anchors + i];
            float w  = output[2 * num_anchors + i];
            float h  = output[3 * num_anchors + i];

            float dw = (input_w - scale * img_w) / 2;
            float dh = (input_h - scale * img_h) / 2;
            
            float x = (cx - w * 0.5f - dw) / scale;
            float y = (cy - h * 0.5f - dh) / scale;
            float w_orig = w / scale;
            float h_orig = h / scale;

            boxes.push_back(cv::Rect(x, y, w_orig, h_orig));
            confidences.push_back(max_cls_score);
            class_ids.push_back(max_cls_id);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, config_.conf_thres, config_.nms_thres, indices);

    objects.clear();
    for (int idx : indices) {
        Object obj;
        obj.rect = boxes[idx];
        obj.label = class_ids[idx];
        obj.prob = confidences[idx];
        objects.push_back(obj);
    }
}