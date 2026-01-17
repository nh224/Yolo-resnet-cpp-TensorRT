#pragma once
#include <string>

// Engine 构建器 - 将 ONNX 转换为 TensorRT Engine
class EngineBuilder {
public:
    // 从 ONNX 文件构建 TensorRT Engine
    static bool build_from_onnx(
        const std::string& onnx_path,
        const std::string& engine_path,
        bool use_fp16 = true
    );
};
