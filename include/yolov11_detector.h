/**
 * @file yolov11_detector.h
 * @brief Header file for YOLOv11 detector and exporter classes.
 *
 * Separates the model export functionality (ONNX to TensorRT Engine conversion)
 * from the detection functionality (inference and postprocessing).
 */

#pragma once

#include "NvInfer.h"
#include <NvOnnxParser.h>
#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using namespace std;
using namespace cv;

/**
 * @struct Detection
 * @brief A structure representing a detected object.
 *
 * Contains the confidence score, class ID, and bounding box for a detected object.
 */
struct Detection
{
    float conf;      //!< Confidence score of the detection.
    int class_id;    //!< Class ID of the detected object.
    Rect bbox;       //!< Bounding box of the detected object.
};

/**
 * @class YOLOv11Exporter
 * @brief A class for exporting YOLOv11 models from ONNX to TensorRT Engine format.
 *
 * This class handles the conversion of ONNX models to optimized TensorRT engines
 * with support for FP16 precision.
 */
class YOLOv11Exporter
{
public:
    /**
     * @brief Constructor to initialize the YOLOv11Exporter.
     *
     * @param onnx_path Path to the ONNX model file.
     * @param logger Reference to a TensorRT logger for error reporting.
     * @param use_fp16 Enable FP16 precision (default: true).
     */
    YOLOv11Exporter(const string& onnx_path, nvinfer1::ILogger& logger, bool use_fp16 = true);

    /**
     * @brief Destructor to clean up resources.
     */
    ~YOLOv11Exporter();

    /**
     * @brief Build the TensorRT engine from the ONNX model.
     *
     * @return True if the engine was built successfully, false otherwise.
     */
    bool build();

    /**
     * @brief Save the TensorRT engine to a file.
     *
     * @param engine_path Path to save the serialized engine. If empty, uses ONNX path with .engine extension.
     * @return True if the engine was saved successfully, false otherwise.
     */
    bool saveEngine(const string& engine_path = "");

    /**
     * @brief Get the input width of the model.
     */
    int getInputWidth() const { return input_w; }

    /**
     * @brief Get the input height of the model.
     */
    int getInputHeight() const { return input_h; }

private:
    string onnx_path;              //!< Path to the ONNX model file.
    bool use_fp16;                 //!< Flag to enable FP16 precision.
    nvinfer1::ILogger& logger;     //!< Reference to TensorRT logger.

    IRuntime* runtime;             //!< TensorRT runtime.
    ICudaEngine* engine;           //!< TensorRT engine.
    INetworkDefinition* network;   //!< Network definition.
    IBuilderConfig* config;        //!< Builder configuration.
    nvonnxparser::IParser* parser; //!< ONNX parser.

    int input_w;                   //!< Width of the input image.
    int input_h;                   //!< Height of the input image.
};

/**
 * @class YOLOv11Detector
 * @brief A class for running YOLOv11 object detection using TensorRT and OpenCV.
 *
 * This class handles model loading from a TensorRT engine, preprocessing,
 * inference, and postprocessing to detect objects in images.
 */
class YOLOv11Detector
{
public:
    /**
     * @brief Constructor to initialize the YOLOv11Detector.
     *
     * Loads the model engine and initializes TensorRT objects.
     *
     * @param engine_path Path to the TensorRT engine file.
     * @param logger Reference to a TensorRT logger for error reporting.
     * @param conf_threshold Confidence threshold for filtering detections (default: 0.3).
     * @param nms_threshold Non-Maximum Suppression threshold (default: 0.4).
     */
    YOLOv11Detector(const string& engine_path, nvinfer1::ILogger& logger,
                    float conf_threshold = 0.3f, float nms_threshold = 0.4f);

    nvinfer1::ILogger& logger;     //!< Reference to TensorRT logger.

    /**
     * @brief Destructor to clean up resources.
     *
     * Frees the allocated memory and TensorRT resources.
     */
    ~YOLOv11Detector();

    /**
     * @brief Preprocess the input image.
     *
     * Prepares the image for inference by resizing and normalizing it.
     *
     * @param image The input image to be preprocessed.
     */
    void preprocess(Mat& image);

    /**
     * @brief Run inference on the preprocessed image.
     *
     * Executes the TensorRT engine for object detection.
     */
    void infer();

    /**
     * @brief Postprocess the output from the model.
     *
     * Filters and decodes the raw output from the TensorRT engine into detection results.
     *
     * @param output A vector to store the detected objects.
     */
    void postprocess(vector<Detection>& output);

    /**
     * @brief Draw the detected objects on the image.
     *
     * Overlays bounding boxes and class labels on the image for visualization.
     *
     * @param image The input image where the detections will be drawn.
     * @param output A vector of detections to be visualized.
     */
    void draw(Mat& image, const vector<Detection>& output);

    /**
     * @brief Get the input width of the model.
     */
    int getInputWidth() const { return input_w; }

    /**
     * @brief Get the input height of the model.
     */
    int getInputHeight() const { return input_h; }

    /**
     * @brief Get the number of classes.
     */
    int getNumClasses() const { return num_classes; }

    /**
     * @brief Set the confidence threshold.
     */
    void setConfThreshold(float threshold) { conf_threshold = threshold; }

    /**
     * @brief Set the NMS threshold.
     */
    void setNMSThreshold(float threshold) { nms_threshold = threshold; }

private:
    /**
     * @brief Initialize TensorRT components from the given engine file.
     *
     * @param engine_path Path to the serialized TensorRT engine file.
     */
    void init(const string& engine_path);

    float* gpu_buffers[2];         //!< Device buffers for engine execution.
    float* cpu_output_buffer;      //!< Pointer to the output buffer on the host.

    cudaStream_t stream;           //!< CUDA stream for asynchronous execution.
    IRuntime* runtime;             //!< TensorRT runtime.
    ICudaEngine* engine;           //!< TensorRT engine.
    IExecutionContext* context;    //!< Execution context.

    // Model parameters
    int input_w;                   //!< Width of the input image.
    int input_h;                   //!< Height of the input image.
    int num_detections;            //!< Number of detections output by the model.
    int detection_attribute_size;  //!< Size of each detection attribute.
    int num_classes;               //!< Number of object classes that can be detected.
    const int MAX_IMAGE_SIZE = 4096 * 4096; //!< Maximum allowed input image size.
    float conf_threshold;          //!< Confidence threshold for filtering detections.
    float nms_threshold;           //!< Non-Maximum Suppression threshold.
};
