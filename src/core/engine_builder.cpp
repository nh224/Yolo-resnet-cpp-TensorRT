#include "core/engine_builder.h"
#include "NvOnnxParser.h"
#include <iostream>
#include <fstream>
#include <memory>

// TensorRT Logger
static class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
} logger;

bool EngineBuilder::build_from_onnx(
    const std::string& onnx_path,
    const std::string& engine_path,
    bool use_fp16
) {
    std::cout << "Building engine from " << onnx_path << "..." << std::endl;

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder) return false;

    // TensorRT 8+: 使用 kEXPLICIT_BATCH flag
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) return false;

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (!parser) return false;

    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX file." << std::endl;
        return false;
    }

    // 打印网络信息用于调试
    std::cout << "Network layers: " << network->getNbLayers() << std::endl;
    std::cout << "Network inputs: " << network->getNbInputs() << std::endl;
    std::cout << "Network outputs: " << network->getNbOutputs() << std::endl;
    for (int i = 0; i < network->getNbInputs(); ++i) {
        auto input = network->getInput(i);
        auto dims = input->getDimensions();
        std::cout << "  Input " << i << ": " << input->getName() << " [";
        for (int j = 0; j < dims.nbDims; ++j) {
            std::cout << (j > 0 ? ", " : "") << dims.d[j];
        }
        std::cout << "]" << std::endl;
    }
    for (int i = 0; i < network->getNbOutputs(); ++i) {
        auto output = network->getOutput(i);
        auto dims = output->getDimensions();
        std::cout << "  Output " << i << ": " << output->getName() << " [";
        for (int j = 0; j < dims.nbDims; ++j) {
            std::cout << (j > 0 ? ", " : "") << dims.d[j];
        }
        std::cout << "]" << std::endl;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) return false;

    // 设置精度约束：强制所有层使用 FP32（除非显式启用 FP16）
    config->setFlag(nvinfer1::BuilderFlag::kTF32);  // 启用 TF32（在 Ampere+ GPU 上）

    if (use_fp16 && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "FP16 mode enabled." << std::endl;

        // 重要: 强制输出层使用 FP32 以保持 Softmax 精度
        // 这样可以避免 FP16 导致的 logits 缩放问题
        for (int i = 0; i < network->getNbOutputs(); ++i) {
            auto output = network->getOutput(i);
            output->setType(nvinfer1::DataType::kFLOAT);
            std::cout << "Output layer '" << output->getName() << "' set to FP32" << std::endl;
        }
    } else {
        std::cout << "FP32 mode enabled (precision)." << std::endl;
    }

    // 禁用某些可能导致数值问题的优化
    // DLA 是深度学习加速器，可能导致数值不稳定
    // config->setDLACore(0);  // 不使用 DLA

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1GB

    // 检查是否有动态输入并设置 optimization profile
    bool has_dynamic_input = false;
    for (int i = 0; i < network->getNbInputs(); ++i) {
        auto input = network->getInput(i);
        const auto dims = input->getDimensions();
        for (int j = 0; j < dims.nbDims; ++j) {
            if (dims.d[j] == -1) {
                has_dynamic_input = true;
                break;
            }
        }
    }

    if (has_dynamic_input) {
        std::cout << "Network has dynamic inputs, setting optimization profile..." << std::endl;

        // 创建 optimization profile
        auto profile = builder->createOptimizationProfile();
        if (!profile) {
            std::cerr << "Failed to create optimization profile." << std::endl;
            return false;
        }

        // 为每个输入设置 shapes
        for (int i = 0; i < network->getNbInputs(); ++i) {
            auto input = network->getInput(i);
            const auto name = input->getName();
            auto dims = input->getDimensions();

            // 设置 min, opt, max shapes
            // 对于 batch 维度 (通常是 dim 0)，设置为 1, 1, 1
            // 对于其他动态维度，也设置为固定值
            nvinfer1::Dims min_dims = dims;
            nvinfer1::Dims opt_dims = dims;
            nvinfer1::Dims max_dims = dims;

            for (int j = 0; j < dims.nbDims; ++j) {
                if (dims.d[j] == -1) {
                    // 动态维度：batch 设为 1，其他设为原始尺寸
                    if (j == 0) {  // batch 维度
                        min_dims.d[j] = 1;
                        opt_dims.d[j] = 1;
                        max_dims.d[j] = 1;
                    } else {  // 其他动态维度
                        min_dims.d[j] = dims.d[j];  // 保持原值或设置合理值
                        opt_dims.d[j] = dims.d[j];
                        max_dims.d[j] = dims.d[j];
                    }
                }
            }

            profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, min_dims);
            profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, opt_dims);
            profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, max_dims);

            std::cout << "  Input " << name << ": set shapes [min/opt/max]" << std::endl;
        }

        config->addOptimizationProfile(profile);
    }

    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan) {
        std::cerr << "Failed to build serialized network." << std::endl;
        return false;
    }

    std::ofstream file(engine_path, std::ios::binary);
    if (!file.good()) return false;
    file.write(reinterpret_cast<const char*>(plan->data()), plan->size());
    file.close();

    std::cout << "Engine saved to: " << engine_path << std::endl;
    return true;
}
