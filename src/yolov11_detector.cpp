#include "yolov11_detector.h"
#include "logging.h"
#include "cuda_utils.h"
#include "macros.h"
#include "preprocess.h"
#include "common.h"
#include <NvOnnxParser.h>
#include <fstream>
#include <iostream>

// External color definitions and class names (defined in common.h or similar)
extern const std::vector<std::vector<int>> COLORS;
extern const std::vector<std::string> CLASS_NAMES;

// ============================================================================
// YOLOv11Exporter Implementation
// ============================================================================

YOLOv11Exporter::YOLOv11Exporter(const string& onnx_path, nvinfer1::ILogger& logger, bool use_fp16)
    : onnx_path(onnx_path), logger(logger), use_fp16(use_fp16)
    , runtime(nullptr), engine(nullptr), network(nullptr), config(nullptr), parser(nullptr)
    , input_w(0), input_h(0)
{
}

YOLOv11Exporter::~YOLOv11Exporter()
{
    // Clean up in reverse order of creation
    delete parser;
    delete config;
    delete network;
    delete engine;
    delete runtime;
}

bool YOLOv11Exporter::build()
{
    // Create a TensorRT builder
    auto builder = createInferBuilder(logger);
    if (!builder) {
        std::cerr << "Failed to create TensorRT builder" << std::endl;
        return false;
    }

#if NV_TENSORRT_MAJOR < 10
    // For TensorRT versions less than 10, use explicit batch flag
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    network = builder->createNetworkV2(explicitBatch);
#else
    // For TensorRT 10+, explicit batch is default
    network = builder->createNetworkV2(0);
#endif

    if (!network) {
        std::cerr << "Failed to create network" << std::endl;
        delete builder;
        return false;
    }

    // Create builder configuration
    config = builder->createBuilderConfig();
    if (!config) {
        std::cerr << "Failed to create builder config" << std::endl;
        delete builder;
        return false;
    }

    // Enable FP16 precision if specified
    if (use_fp16) {
        config->setFlag(BuilderFlag::kFP16);
    }

    // Create an ONNX parser
    parser = nvonnxparser::createParser(*network, logger);
    if (!parser) {
        std::cerr << "Failed to create ONNX parser" << std::endl;
        delete builder;
        return false;
    }

    // Parse the ONNX model file
    bool parsed = parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
    if (!parsed) {
        std::cerr << "Failed to parse ONNX model: " << onnx_path << std::endl;
        delete builder;
        return false;
    }

    // Build the serialized network plan
    nvinfer1::IHostMemory* plan = builder->buildSerializedNetwork(*network, *config);
    if (!plan) {
        std::cerr << "Failed to build serialized network" << std::endl;
        delete builder;
        return false;
    }

    // Create a TensorRT runtime
    runtime = createInferRuntime(logger);
    if (!runtime) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        delete plan;
        delete builder;
        return false;
    }

    // Deserialize the CUDA engine from the serialized plan
    engine = runtime->deserializeCudaEngine(plan->data(), plan->size());
    if (!engine) {
        std::cerr << "Failed to deserialize CUDA engine" << std::endl;
        delete plan;
        delete builder;
        return false;
    }

    // Retrieve input dimensions from the engine
#if NV_TENSORRT_MAJOR < 10
    auto input_dims = engine->getBindingDimensions(0);
    input_w = input_dims.d[3];
    input_h = input_dims.d[2];
#else
    auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
    input_w = input_dims.d[3];
    input_h = input_dims.d[2];
#endif

    delete plan;
    delete builder;

    std::cout << "Engine built successfully. Input size: " << input_w << "x" << input_h << std::endl;
    return true;
}

bool YOLOv11Exporter::saveEngine(const string& engine_path)
{
    if (!engine) {
        std::cerr << "No engine to save. Please call build() first." << std::endl;
        return false;
    }

    // Generate the engine file path
    std::string output_path = engine_path;
    if (output_path.empty()) {
        size_t dotIndex = onnx_path.find_last_of(".");
        if (dotIndex != std::string::npos) {
            output_path = onnx_path.substr(0, dotIndex) + ".engine";
        } else {
            output_path = onnx_path + ".engine";
        }
    }

    // Serialize the engine
    nvinfer1::IHostMemory* data = engine->serialize();
    if (!data) {
        std::cerr << "Failed to serialize engine" << std::endl;
        return false;
    }

    // Write the serialized engine to file
    std::ofstream file(output_path, std::ios::binary | std::ios::out);
    if (!file.is_open()) {
        std::cerr << "Failed to create engine file: " << output_path << std::endl;
        delete data;
        return false;
    }

    file.write((const char*)data->data(), data->size());
    file.close();
    delete data;

    std::cout << "Engine saved to: " << output_path << std::endl;
    return true;
}

// ============================================================================
// YOLOv11Detector Implementation
// ============================================================================

YOLOv11Detector::YOLOv11Detector(const string& engine_path, nvinfer1::ILogger& logger,
                                 float conf_threshold, float nms_threshold)
    : runtime(nullptr), engine(nullptr), context(nullptr), cpu_output_buffer(nullptr), stream(0)
    , conf_threshold(conf_threshold), nms_threshold(nms_threshold), logger(logger)
{
    // Initialize GPU buffers to nullptr
    gpu_buffers[0] = nullptr;
    gpu_buffers[1] = nullptr;

    // Initialize from engine file
    init(engine_path);
}

YOLOv11Detector::~YOLOv11Detector()
{
    // Destroy CUDA preprocessing resources
    cuda_preprocess_destroy();

    // Synchronize and destroy the CUDA stream
    if (stream != 0) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    // Free allocated GPU buffers
    for (int i = 0; i < 2; i++) {
        if (gpu_buffers[i] != nullptr) {
            CUDA_CHECK(cudaFree(gpu_buffers[i]));
            gpu_buffers[i] = nullptr;
        }
    }

    // Free CPU output buffer
    if (cpu_output_buffer != nullptr) {
        delete[] cpu_output_buffer;
        cpu_output_buffer = nullptr;
    }

    // Delete TensorRT objects
    delete context;
    delete engine;
    delete runtime;
}

void YOLOv11Detector::init(const string& engine_path)
{
    // Open the engine file in binary mode
    std::ifstream engineStream(engine_path, std::ios::binary);
    if (!engineStream.is_open()) {
        throw std::runtime_error("Failed to open engine file: " + engine_path);
    }

    // Get file size
    engineStream.seekg(0, std::ios::end);
    const size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, std::ios::beg);

    // Allocate memory and read engine data
    std::unique_ptr<char[]> engineData(new char[modelSize]);
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();

    // Create TensorRT runtime
    runtime = createInferRuntime(logger);
    if (!runtime) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    // Deserialize the CUDA engine
    engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
    if (!engine) {
        throw std::runtime_error("Failed to deserialize CUDA engine");
    }

    // Create execution context
    context = engine->createExecutionContext();
    if (!context) {
        throw std::runtime_error("Failed to create execution context");
    }

    // Retrieve input/output dimensions
#if NV_TENSORRT_MAJOR < 10
    auto input_dims = engine->getBindingDimensions(0);
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];

    auto output_dims = engine->getBindingDimensions(1);
    detection_attribute_size = output_dims.d[1];
    num_detections = output_dims.d[2];
#else
    auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];

    auto output_dims = engine->getTensorShape(engine->getIOTensorName(1));
    detection_attribute_size = output_dims.d[1];
    num_detections = output_dims.d[2];
#endif

    // Calculate number of classes (detection attributes - 4 for bbox coordinates)
    num_classes = detection_attribute_size - 4;

    // Allocate CPU memory for output buffer
    cpu_output_buffer = new float[detection_attribute_size * num_detections];

    // Allocate GPU memory for input and output buffers
    CUDA_CHECK(cudaMalloc(&gpu_buffers[0], 3 * input_w * input_h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_buffers[1], detection_attribute_size * num_detections * sizeof(float)));

#if NV_TENSORRT_MAJOR >= 10
    // Set tensor addresses for TensorRT 10+ (must be AFTER cudaMalloc!)
    context->setTensorAddress(engine->getIOTensorName(0), gpu_buffers[0]);
    context->setTensorAddress(engine->getIOTensorName(1), gpu_buffers[1]);
#endif

    // Initialize CUDA preprocessing
    cuda_preprocess_init(MAX_IMAGE_SIZE);

    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::cout << "Detector initialized. Input size: " << input_w << "x" << input_h
              << ", Classes: " << num_classes << std::endl;
}

void YOLOv11Detector::preprocess(Mat& image)
{
    // Perform CUDA-based preprocessing
    cuda_preprocess(image.ptr(), image.cols, image.rows, gpu_buffers[0], input_w, input_h, stream);
    // Synchronize to ensure preprocessing is complete
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void YOLOv11Detector::infer()
{
#if NV_TENSORRT_MAJOR < 10
    // For TensorRT versions less than 10, use enqueueV2
    context->enqueueV2((void**)gpu_buffers, stream, nullptr);
#else
    // For TensorRT 10+, use enqueueV3
    context->enqueueV3(stream);
#endif
}

void YOLOv11Detector::postprocess(vector<Detection>& output)
{
    // Asynchronously copy output from GPU to CPU
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[1],
                              num_detections * detection_attribute_size * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    vector<Rect> boxes;
    vector<int> class_ids;
    vector<float> confidences;

    // Create a matrix view of the detection output
    const Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);

    // Iterate over each detection
    for (int i = 0; i < det_output.cols; ++i) {
        // Extract class scores for the current detection
        const Mat classes_scores = det_output.col(i).rowRange(4, 4 + num_classes);
        Point class_id_point;
        double score;

        // Find the class with the maximum score
        minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        // Check if the confidence score exceeds the threshold
        if (score > conf_threshold) {
            // Extract bounding box coordinates (center x, center y, width, height)
            const float cx = det_output.at<float>(0, i);
            const float cy = det_output.at<float>(1, i);
            const float ow = det_output.at<float>(2, i);
            const float oh = det_output.at<float>(3, i);

            // Convert to top-left corner format
            Rect box;
            box.x = static_cast<int>(cx - 0.5f * ow);
            box.y = static_cast<int>(cy - 0.5f * oh);
            box.width = static_cast<int>(ow);
            box.height = static_cast<int>(oh);

            boxes.push_back(box);
            class_ids.push_back(class_id_point.y);
            confidences.push_back(score);
        }
    }

    // Apply Non-Maximum Suppression to remove overlapping boxes
    vector<int> nms_result;
    dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);

    // Populate output detections
    for (int idx : nms_result) {
        Detection result;
        result.class_id = class_ids[idx];
        result.conf = confidences[idx];
        result.bbox = boxes[idx];
        output.push_back(result);
    }
}

void YOLOv11Detector::draw(Mat& image, const vector<Detection>& output)
{
    // Calculate scaling ratios
    const float ratio_h = input_h / static_cast<float>(image.rows);
    const float ratio_w = input_w / static_cast<float>(image.cols);

    for (const auto& detection : output) {
        Rect box = detection.bbox;
        const int class_id = detection.class_id;
        const float conf = detection.conf;

        // Get color for this class
        cv::Scalar color = cv::Scalar(COLORS[class_id][0], COLORS[class_id][1], COLORS[class_id][2]);

        // Adjust bounding box coordinates based on aspect ratio (letterbox/pad)
        if (ratio_h > ratio_w) {
            // Width is the limiting factor
            box.x = box.x / ratio_w;
            box.y = (box.y - (input_h - ratio_w * image.rows) / 2) / ratio_w;
            box.width = box.width / ratio_w;
            box.height = box.height / ratio_w;
        } else {
            // Height is the limiting factor
            box.x = (box.x - (input_w - ratio_h * image.cols) / 2) / ratio_h;
            box.y = box.y / ratio_h;
            box.width = box.width / ratio_h;
            box.height = box.height / ratio_h;
        }

        // Draw the bounding box
        rectangle(image, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), color, 3);

        // Prepare the label text
        std::string class_string = CLASS_NAMES[class_id] + ' ' + std::to_string(conf).substr(0, 4);

        // Calculate text size for background rectangle
        Size text_size = getTextSize(class_string, FONT_HERSHEY_DUPLEX, 1, 2, 0);
        Rect text_rect(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);

        // Draw background rectangle for text
        rectangle(image, text_rect, color, FILLED);

        // Draw the text label
        putText(image, class_string, Point(box.x + 5, box.y - 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2, 0);
    }
}
