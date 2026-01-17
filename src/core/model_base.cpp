#include "core/model_base.h"
#include "preprocess.h"
#include "postprocess.h"
#include <fstream>
#include <iostream>
#include <algorithm>

static class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
} logger;

ModelBase::ModelBase(const std::string& engine_path, bool is_segmentation)
    : is_segmentation_(is_segmentation) {
    gpu_buffers.clear();
    gpu_buffers.resize(3, nullptr);  // 最多3个buffer: input, mask_proto, bbox_output
    cpu_output_buffer = nullptr;
    cpu_mask_buffer = nullptr;

    init_context(engine_path);
    cuda_preprocess_init(4096 * 4096);
    cuda_postprocess_init();  // 初始化后处理模块
}

ModelBase::~ModelBase() {
    cuda_preprocess_destroy();
    cuda_postprocess_destroy();  // 清理后处理模块

    if (stream) cudaStreamDestroy(stream);
    for (auto& buf : gpu_buffers) {
        if (buf) cudaFree(buf);
    }
    if (cpu_output_buffer) cudaFreeHost(cpu_output_buffer);
    if (cpu_mask_buffer) cudaFreeHost(cpu_mask_buffer);

    delete context;
    delete engine;
    delete runtime;
}

void ModelBase::init_context(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error: Read engine failed: " << engine_path << std::endl;
        exit(1);
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    runtime = nvinfer1::createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engineData.data(), size);
    context = engine->createExecutionContext();
    cudaStreamCreate(&stream);

    int nbIOTensors = engine->getNbIOTensors();
    int input_count = 0;
    int output_count = 0;

    for (int i = 0; i < nbIOTensors; ++i) {
        const char* name = engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = engine->getTensorIOMode(name);
        nvinfer1::Dims dims = engine->getTensorShape(name);

        int vol = 1;
        for(int j=0; j<dims.nbDims; ++j) {
            if(dims.d[j] > 0) vol *= dims.d[j];
        }

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_tensor_name = name;
            input_size = vol;
            if(dims.nbDims == 4) { input_h = dims.d[2]; input_w = dims.d[3]; }
            else { input_h = dims.d[1]; input_w = dims.d[2]; }

            if (input_h <= 0) {
                input_h = 640;
                input_w = 640;
            }
            input_count++;
        } else {
            output_tensor_names.push_back(name);
            output_sizes.push_back(vol);

            // 分割模型：根据输出名称判断是 bbox 还是 mask_proto
            if (is_segmentation_) {
                // YOLOv8-seg: output0=bbox [1,116,8400] or [1,8400,116], output1=mask_proto [1,32,160,160]
                std::string name_str(name);
                // 判断依据：名称包含"output0" 或者维度为3 (bbox格式)
                if (name_str.find("output0") != std::string::npos || dims.nbDims == 3) {
                    // bbox 输出
                    output_size = vol;
                    if (dims.nbDims == 3 && dims.d[0] == 1) {
                        // 检查是 [1, 116, 8400] 还是 [1, 8400, 116]
                        // 通过比较维度大小判断：116 < 8400
                        if (dims.d[1] < dims.d[2]) {
                            // [1, 116, 8400] - TensorRT/ONNX 格式
                            detection_attribute_size = dims.d[1];  // 116
                            num_detections = dims.d[2];            // 8400
                        } else {
                            // [1, 8400, 116] - 期望格式
                            detection_attribute_size = dims.d[2];  // 116
                            num_detections = dims.d[1];            // 8400
                        }
                    } else {
                        detection_attribute_size = dims.d[dims.nbDims - 2];
                        num_detections = dims.d[dims.nbDims - 1];
                    }
                }
                // mask_proto 输出由 SegmentationModel 处理
            } else {
                // 普通检测/分类模型
                output_size = vol;
                if (dims.nbDims >= 2) {
                    detection_attribute_size = dims.d[dims.nbDims - 2];
                    num_detections = dims.d[dims.nbDims - 1];
                }
            }
            output_count++;
        }
    }

    // 分配 GPU 缓冲区
    gpu_buffers[0] = nullptr;  // 输入
    cudaMalloc(&gpu_buffers[0], input_size * sizeof(float));

    if (is_segmentation_) {
        // 分割模型：0=input, 1=bbox, 2=mask_proto
        cudaMalloc(&gpu_buffers[1], output_sizes[0] * sizeof(float));  // bbox
        cudaMalloc(&gpu_buffers[2], output_sizes[1] * sizeof(float));  // mask_proto
        cudaMallocHost((void**)&cpu_output_buffer, output_sizes[0] * sizeof(float));  // bbox
        cudaMallocHost((void**)&cpu_mask_buffer, output_sizes[1] * sizeof(float));   // mask_proto
    } else {
        // 普通模型：0=input, 1=output
        cudaMalloc(&gpu_buffers[1], output_size * sizeof(float));
        cudaMallocHost((void**)&cpu_output_buffer, output_size * sizeof(float));
    }

    // ★ 优化：初始化时设定好 Input Shape 和 Tensor Address
    nvinfer1::Dims input_dims;
    input_dims.nbDims = 4;
    input_dims.d[0] = 1; input_dims.d[1] = 3; input_dims.d[2] = input_h; input_dims.d[3] = input_w;
    context->setInputShape(input_tensor_name.c_str(), input_dims);
    context->setInputTensorAddress(input_tensor_name.c_str(), gpu_buffers[0]);

    // 设置输出 Tensor 地址
    if (is_segmentation_) {
        // 实际引擎：output0=bbox, output1=mask_proto
        context->setTensorAddress(output_tensor_names[0].c_str(), gpu_buffers[1]);  // bbox
        context->setTensorAddress(output_tensor_names[1].c_str(), gpu_buffers[2]);  // mask_proto
    } else {
        context->setTensorAddress(output_tensor_names[0].c_str(), gpu_buffers[1]);
    }
}

void ModelBase::get_mask_dims(int& coeff_len, int& proto_h, int& proto_w) const {
    coeff_len = 0;
    proto_h = 0;
    proto_w = 0;

    if (!is_segmentation_ || engine == nullptr) return;

    // 查找 mask_proto 输出
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = engine->getTensorIOMode(name);

        if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            std::string name_str(name);
            // 如果名称不包含 output0，则为 mask_proto
            if (name_str.find("output0") == std::string::npos) {
                nvinfer1::Dims dims = engine->getTensorShape(name);
                if (dims.nbDims == 4) {
                    coeff_len = dims.d[1];
                    proto_h = dims.d[2];
                    proto_w = dims.d[3];
                    return;
                }
            }
        }
    }
}
