#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "NvOnnxParser.h" // 用于模型转换
#include "wrapper.h"      // 推理封装

// TensorRT Logger
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // 只打印警告及以上信息
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
} logger;

// ======================================================================================
// 辅助函数：构建 Engine (ONNX -> TensorRT Engine) [适配 TensorRT 10]
// ======================================================================================
bool build_engine(const std::string& onnx_path, const std::string& engine_path) {
    std::cout << "Building engine from " << onnx_path << "..." << std::endl;

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder) return false;

    // --- TensorRT 10 修改点 ---
    // TensorRT 10 默认就是 Explicit Batch，旧的 kEXPLICIT_BATCH flag 已废弃
    // 直接传 0 即可
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    // -------------------------
    
    if (!network) return false;

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (!parser) return false;

    // 解析 ONNX
    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX file." << std::endl;
        return false;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) return false;

    // 启用 FP16 (如果支持)
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "FP16 mode enabled." << std::endl;
    }

    // 设置最大工作空间 (1GB)
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30); 

    // 序列化 Engine
    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan) {
        std::cerr << "Failed to build serialized network." << std::endl;
        return false;
    }

    // 保存文件
    std::ofstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to open engine file for writing." << std::endl;
        return false;
    }
    file.write(reinterpret_cast<const char*>(plan->data()), plan->size());
    file.close();

    std::cout << "Engine built successfully and saved to: " << engine_path << std::endl;
    return true;
}

// ======================================================================================
// Main Function
// ======================================================================================
int main(int argc, char* argv[]) {

    // 颜色代码
    const std::string RED = "\033[31m";
    const std::string GREEN = "\033[32m";
    const std::string RESET = "\033[0m";

    if (argc < 2) {
        std::cerr << RED << "Usage: " << RESET << std::endl;
        std::cerr << "  1. Convert: " << argv[0] << " convert <onnx_path> <engine_path>" << std::endl;
        std::cerr << "  2. Infer:   " << argv[0] << " <mode> <model_type> <input_path> <engine_path>" << std::endl;
        return 1;
    }

    std::string mode = argv[1];

    // -------------------------------------------------------------------------
    // Mode: Convert (ONNX -> Engine)
    // -------------------------------------------------------------------------
    if (mode == "convert") {
        if (argc != 4) {
            std::cerr << RED << "Usage: " << argv[0] << " convert <onnx_path> <engine_path>" << RESET << std::endl;
            return 1;
        }
        std::string onnxPath = argv[2];
        std::string enginePath = argv[3];

        if (!build_engine(onnxPath, enginePath)) {
            std::cerr << RED << "Conversion failed!" << RESET << std::endl;
            return 1;
        }
        return 0;
    }

    // -------------------------------------------------------------------------
    // Mode: Inference (Image/Video)
    // -------------------------------------------------------------------------
    if (mode == "infer_video" || mode == "infer_image") {
        if (argc != 5) {
            std::cerr << RED << "Usage: " << argv[0] << " " << mode << " <model_type> <input_path> <engine_path>" << RESET << std::endl;
            std::cerr << "  <model_type>: 'yolo' or 'resnet'" << std::endl;
            return 1;
        }

        std::string typeStr = argv[2];
        std::string inputPath = argv[3];
        std::string enginePath = argv[4];

        // 配置参数
        RunConfig config;
        config.engine_file_path = enginePath;

        if (typeStr == "yolo") {
            config.type = ModelType::YOLO_DETECT;
            config.num_classes = 80; 
        }
        else if (typeStr == "resnet") {
            config.type = ModelType::RESNET_CLS;
            config.num_classes = 1000;
        }
        else {
            std::cerr << RED << "Invalid model type. Use 'yolo' or 'resnet'." << RESET << std::endl;
            return 1;
        }

        try {
            // 初始化推理包装器
            Wrapper model(config);

            // --- 视频推理 ---
            if (mode == "infer_video") {
                cv::VideoCapture cap(inputPath);
                if (!cap.isOpened()) {
                    std::cerr << RED << "Failed to open video." << RESET << std::endl;
                    return 1;
                }
                
                std::string outputVideoPath = "output_video.avi";
                int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
                int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
                cv::VideoWriter video(outputVideoPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(w, h));
                
                cv::Mat frame;
                std::cout << GREEN << "Processing video..." << RESET << std::endl;
                
                while (cap.read(frame)) {
                    if (frame.empty()) break;
                    
                    std::vector<Object> objects;
                    int cls_id = -1; 
                    float cls_score = 0.0f;
                    
                    model.infer(frame, objects, cls_id, cls_score);
                    
                    // 可视化
                    if (config.type == ModelType::YOLO_DETECT) {
                        for(auto& obj : objects) {
                            cv::rectangle(frame, obj.rect, cv::Scalar(0,255,0), 2);
                            std::string label = std::to_string(obj.label) + " " + std::to_string(obj.prob).substr(0,4);
                            cv::putText(frame, label, cv::Point(obj.rect.x, obj.rect.y-5), 0, 0.5, cv::Scalar(0,255,0), 2);
                        }
                    } else {
                        std::string label = "Class: " + std::to_string(cls_id) + " | Score: " + std::to_string(cls_score);
                        cv::putText(frame, label, cv::Point(20,50), 0, 1.0, cv::Scalar(0,0,255), 2);
                    }
                    video.write(frame);
                }
                std::cout << GREEN << "Video saved to " << outputVideoPath << RESET << std::endl;
            }
            // --- 图片推理 ---
            else {
                cv::Mat image = cv::imread(inputPath);
                if (image.empty()) {
                    std::cerr << RED << "Image not found." << RESET << std::endl; 
                    return 1;
                }

                std::vector<Object> objects;
                int cls_id = -1; 
                float cls_score = 0.0f;

                auto start = std::chrono::high_resolution_clock::now();
                
                // 执行推理
                model.infer(image, objects, cls_id, cls_score);
                
                auto end = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(end - start).count();

                // 可视化
                if (config.type == ModelType::YOLO_DETECT) {
                    for(auto& obj : objects) {
                        cv::rectangle(image, obj.rect, cv::Scalar(0,255,0), 2);
                        std::string label = std::to_string(obj.label) + ": " + std::to_string(obj.prob).substr(0,4);
                        cv::putText(image, label, cv::Point(obj.rect.x, obj.rect.y-5), 0, 0.5, cv::Scalar(0,255,0), 2);
                    }
                    std::cout << GREEN << "Detected " << objects.size() << " objects." << RESET << std::endl;
                } else {
                    std::string label = "Class ID: " + std::to_string(cls_id) + " (Score: " + std::to_string(cls_score) + ")";
                    cv::putText(image, label, cv::Point(20,50), 0, 1.0, cv::Scalar(0,0,255), 2);
                    std::cout << GREEN << label << RESET << std::endl;
                }

                cv::imwrite("output.jpg", image);
                std::cout << GREEN << "Latency: " << ms << " ms. Result saved to output.jpg" << RESET << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cerr << RED << "Error: " << e.what() << RESET << std::endl;
            return 1;
        }
    }
    else {
        std::cerr << RED << "Invalid mode. Use 'convert', 'infer_video' or 'infer_image'" << RESET << std::endl;
        return 1;
    }

    return 0;
}