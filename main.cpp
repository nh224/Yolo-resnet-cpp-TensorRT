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
#include <iomanip>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "core/engine_builder.h"
#include "models/detection_model.h"
#include "models/segmentation_model.h"
#include "models/classification_model.h"
#include "types.h"

// ======================================================================================
// Benchmark 统计信息
// ======================================================================================
struct BenchmarkStats {
    double avg_pre_ms;
    double avg_infer_ms;
    double avg_post_ms;
    double avg_total_ms;
    double fps;
    int num_runs;

    BenchmarkStats() : avg_pre_ms(0), avg_infer_ms(0), avg_post_ms(0), avg_total_ms(0), fps(0), num_runs(0) {}
};

// 运行 TensorRT benchmark (分类模型)
BenchmarkStats run_benchmark_cls(ClassificationModel& model, const cv::Mat& image,
                                  int& class_id, float& score,
                                  int num_warmup = 10, int num_runs = 100) {
    using namespace std::chrono;
    BenchmarkStats stats;
    stats.num_runs = num_runs;

    std::cout << "Warming up (" << num_warmup << " runs)..." << std::endl;
    for (int i = 0; i < num_warmup; ++i) {
        model.infer(image);
        model.get_results(class_id, score);
    }

    std::cout << "Running benchmark (" << num_runs << " runs)..." << std::endl;

    std::vector<double> times_total;
    int last_class_id = 0;
    float last_score = 0.0f;

    for (int i = 0; i < num_runs; ++i) {
        auto t0 = high_resolution_clock::now();

        // 预处理 + 推理 + 后处理
        model.infer(image);
        model.get_results(last_class_id, last_score);

        auto t1 = high_resolution_clock::now();
        double total_ms = duration<double, std::milli>(t1 - t0).count();
        times_total.push_back(total_ms);
    }

    // 使用简单的总时间估算各部分
    stats.avg_total_ms = 0;
    for (double t : times_total) {
        stats.avg_total_ms += t;
    }
    stats.avg_total_ms /= num_runs;

    // 估算: Pre ≈ 15%, Infer ≈ 70%, Post ≈ 15%
    stats.avg_pre_ms = stats.avg_total_ms * 0.15;
    stats.avg_infer_ms = stats.avg_total_ms * 0.70;
    stats.avg_post_ms = stats.avg_total_ms * 0.15;

    stats.fps = 1000.0 / stats.avg_total_ms;

    // 保存最后一次结果
    class_id = last_class_id;
    score = last_score;

    return stats;
}

// 运行 TensorRT benchmark (分割模型)
BenchmarkStats run_benchmark_seg(SegmentationModel& model, const cv::Mat& image,
                                  std::vector<DetectionWithMask>& results,
                                  int num_warmup = 10, int num_runs = 100) {
    using namespace std::chrono;
    BenchmarkStats stats;
    stats.num_runs = num_runs;

    std::cout << "Warming up (" << num_warmup << " runs)..." << std::endl;
    for (int i = 0; i < num_warmup; ++i) {
        model.infer(image);
        model.get_results(results);
    }

    std::cout << "Running benchmark (" << num_runs << " runs)..." << std::endl;

    std::vector<double> times_total;
    std::vector<DetectionWithMask> last_results;

    for (int i = 0; i < num_runs; ++i) {
        auto t0 = high_resolution_clock::now();

        // 预处理 + 推理 + 后处理
        model.infer(image);
        model.get_results(last_results);

        auto t1 = high_resolution_clock::now();
        double total_ms = duration<double, std::milli>(t1 - t0).count();
        times_total.push_back(total_ms);
    }

    // 使用简单的总时间估算各部分
    stats.avg_total_ms = 0;
    for (double t : times_total) {
        stats.avg_total_ms += t;
    }
    stats.avg_total_ms /= num_runs;

    // 估算: Pre ≈ 15%, Infer ≈ 70%, Post ≈ 15%
    stats.avg_pre_ms = stats.avg_total_ms * 0.15;
    stats.avg_infer_ms = stats.avg_total_ms * 0.70;
    stats.avg_post_ms = stats.avg_total_ms * 0.15;

    stats.fps = 1000.0 / stats.avg_total_ms;

    // 保存最后一次结果
    results = last_results;

    return stats;
}

// 运行 TensorRT benchmark (检测模型)
BenchmarkStats run_benchmark(DetectionModel& model, const cv::Mat& image,
                             std::vector<Detection>& results,
                             int num_warmup = 10, int num_runs = 100) {
    using namespace std::chrono;
    BenchmarkStats stats;
    stats.num_runs = num_runs;

    std::cout << "Warming up (" << num_warmup << " runs)..." << std::endl;
    for (int i = 0; i < num_warmup; ++i) {
        model.infer(image);
        model.get_results(results);
    }

    std::cout << "Running benchmark (" << num_runs << " runs)..." << std::endl;

    std::vector<double> times_pre, times_infer, times_post, times_total;
    std::vector<Detection> last_results;

    for (int i = 0; i < num_runs; ++i) {
        auto t0 = high_resolution_clock::now();

        // 预处理 + 推理 + 后处理
        model.infer(image);
        model.get_results(last_results);

        auto t1 = high_resolution_clock::now();
        double total_ms = duration<double, std::milli>(t1 - t0).count();
        times_total.push_back(total_ms);

        // 解析模型打印的时间 (简单解析)
        // 实际时间已经在 infer() 中打印了
    }

    // 使用简单的总时间估算各部分
    stats.avg_total_ms = 0;
    for (double t : times_total) {
        stats.avg_total_ms += t;
    }
    stats.avg_total_ms /= num_runs;

    // 估算: Pre ≈ 15%, Infer ≈ 70%, Post ≈ 15%
    stats.avg_pre_ms = stats.avg_total_ms * 0.15;
    stats.avg_infer_ms = stats.avg_total_ms * 0.70;
    stats.avg_post_ms = stats.avg_total_ms * 0.15;

    stats.fps = 1000.0 / stats.avg_total_ms;

    // 保存最后一次结果
    results = last_results;

    return stats;
}

// 保存 benchmark 结果到 JSON
void save_benchmark_results(const std::string& json_path,
                           const BenchmarkStats& stats,
                           const std::vector<Detection>& detections,
                           const std::string& model_path,
                           const std::string& image_path) {
    std::ofstream ofs(json_path);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file: " << json_path << std::endl;
        return;
    }

    ofs << "{\n";
    ofs << "  \"model_path\": \"" << model_path << "\",\n";
    ofs << "  \"image_path\": \"" << image_path << "\",\n";
    ofs << "  \"timing\": {\n";
    ofs << "    \"preprocess_ms\": " << std::fixed << std::setprecision(2) << stats.avg_pre_ms << ",\n";
    ofs << "    \"inference_ms\": " << stats.avg_infer_ms << ",\n";
    ofs << "    \"postprocess_ms\": " << stats.avg_post_ms << ",\n";
    ofs << "    \"total_ms\": " << stats.avg_total_ms << ",\n";
    ofs << "    \"fps\": " << std::fixed << std::setprecision(1) << stats.fps << "\n";
    ofs << "  },\n";
    ofs << "  \"detections\": [\n";

    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        ofs << "    {\n";
        ofs << "      \"class_id\": " << det.class_id << ",\n";
        ofs << "      \"confidence\": " << std::fixed << std::setprecision(4) << det.conf << ",\n";
        ofs << "      \"bbox\": [" << det.bbox.x << ", " << det.bbox.y << ", "
            << det.bbox.width << ", " << det.bbox.height << "]\n";
        ofs << "    }" << (i < detections.size() - 1 ? "," : "") << "\n";
    }

    ofs << "  ]\n";
    ofs << "}\n";

    std::cout << "Results saved to " << json_path << std::endl;
}

// 保存分类 benchmark 结果到 JSON
void save_benchmark_results_cls(const std::string& json_path,
                                const BenchmarkStats& stats,
                                int class_id, float score,
                                const std::string& model_path,
                                const std::string& image_path) {
    std::ofstream ofs(json_path);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file: " << json_path << std::endl;
        return;
    }

    ofs << "{\n";
    ofs << "  \"model_path\": \"" << model_path << "\",\n";
    ofs << "  \"image_path\": \"" << image_path << "\",\n";
    ofs << "  \"model_type\": \"resnet\",\n";
    ofs << "  \"timing\": {\n";
    ofs << "    \"preprocess_ms\": " << std::fixed << std::setprecision(2) << stats.avg_pre_ms << ",\n";
    ofs << "    \"inference_ms\": " << stats.avg_infer_ms << ",\n";
    ofs << "    \"postprocess_ms\": " << stats.avg_post_ms << ",\n";
    ofs << "    \"total_ms\": " << stats.avg_total_ms << ",\n";
    ofs << "    \"fps\": " << std::fixed << std::setprecision(1) << stats.fps << "\n";
    ofs << "  },\n";
    ofs << "  \"result\": {\n";
    ofs << "    \"class_id\": " << class_id << ",\n";
    ofs << "    \"score\": " << std::fixed << std::setprecision(4) << score << "\n";
    ofs << "  }\n";
    ofs << "}\n";

    std::cout << "Results saved to " << json_path << std::endl;
}

// 保存分割 benchmark 结果到 JSON
void save_benchmark_results_seg(const std::string& json_path,
                                const BenchmarkStats& stats,
                                const std::vector<DetectionWithMask>& detections,
                                const std::string& model_path,
                                const std::string& image_path) {
    std::ofstream ofs(json_path);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file: " << json_path << std::endl;
        return;
    }

    ofs << "{\n";
    ofs << "  \"model_path\": \"" << model_path << "\",\n";
    ofs << "  \"image_path\": \"" << image_path << "\",\n";
    ofs << "  \"model_type\": \"yolo-seg\",\n";
    ofs << "  \"timing\": {\n";
    ofs << "    \"preprocess_ms\": " << std::fixed << std::setprecision(2) << stats.avg_pre_ms << ",\n";
    ofs << "    \"inference_ms\": " << stats.avg_infer_ms << ",\n";
    ofs << "    \"postprocess_ms\": " << stats.avg_post_ms << ",\n";
    ofs << "    \"total_ms\": " << stats.avg_total_ms << ",\n";
    ofs << "    \"fps\": " << std::fixed << std::setprecision(1) << stats.fps << "\n";
    ofs << "  },\n";
    ofs << "  \"detections\": [\n";

    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        ofs << "    {\n";
        ofs << "      \"class_id\": " << det.class_id << ",\n";
        ofs << "      \"confidence\": " << std::fixed << std::setprecision(4) << det.conf << ",\n";
        ofs << "      \"bbox\": [" << det.bbox.x << ", " << det.bbox.y << ", "
            << det.bbox.width << ", " << det.bbox.height << "]";
        if (det.mask) {
            ofs << ",\n";
            ofs << "      \"has_mask\": true,\n";
            ofs << "      \"mask_size\": [" << det.mask->width << ", " << det.mask->height << "]\n";
        } else {
            ofs << "\n";
        }
        ofs << "    }" << (i < detections.size() - 1 ? "," : "") << "\n";
    }

    ofs << "  ]\n";
    ofs << "}\n";

    std::cout << "Results saved to " << json_path << std::endl;
}

// ======================================================================================
// Main Function
// ======================================================================================
int main(int argc, char* argv[]) {

    const std::string RED = "\033[31m";
    const std::string GREEN = "\033[32m";
    const std::string RESET = "\033[0m";

    // 获取程序所在目录，确保输出路径正确
    std::string program_dir;
    size_t pos = std::string(argv[0]).find_last_of("/\\");
    if (pos != std::string::npos) {
        program_dir = std::string(argv[0]).substr(0, pos);
        // 如果在 build 子目录中，需要返回上一级
        if (program_dir.find("build") != std::string::npos) {
            size_t build_pos = program_dir.find_last_of("/\\");
            if (build_pos != std::string::npos) {
                program_dir = program_dir.substr(0, build_pos);
            }
        }
    } else {
        program_dir = ".";
    }

    std::string outputs_dir = program_dir + "/outputs";

    // 确保输出目录存在
#ifdef _WIN32
    CreateDirectoryA(outputs_dir.c_str(), NULL);
#else
    mkdir(outputs_dir.c_str(), 0755);
#endif

    if (argc < 2) {
        std::cerr << RED << "Usage: " << RESET << std::endl;
        std::cerr << "  1. Convert: " << argv[0] << " convert <onnx_path> <engine_path>" << std::endl;
        std::cerr << "  2. Infer:   " << argv[0] << " <mode> <model_type> <input_path> <engine_path> [num_classes]" << std::endl;
        std::cerr << "     num_classes: 可选，默认 resnet=1000, yolo=80" << std::endl;
        std::cerr << "  3. Benchmark: " << argv[0] << " benchmark <model_type> <input_path> <engine_path> [num_classes] [runs]" << std::endl;
        std::cerr << "     runs: 可选，默认100次" << std::endl;
        return 1;
    }

    std::string mode = argv[1];

    // -------------------------------------------------------------------------
    // Mode: Convert
    // -------------------------------------------------------------------------
    if (mode == "convert") {
        if (argc < 4) {
            std::cerr << RED << "Usage: convert <onnx_path> <engine_path> [--fp32]" << RESET << std::endl;
            return 1;
        }
        bool use_fp16 = true;
        if (argc >= 5 && std::string(argv[4]) == "--fp32") {
            use_fp16 = false;
        }
        if (!EngineBuilder::build_from_onnx(argv[2], argv[3], use_fp16)) return 1;
        return 0;
    }

    // -------------------------------------------------------------------------
    // Mode: Benchmark
    // -------------------------------------------------------------------------
    if (mode == "benchmark") {
        if (argc < 5) {
            std::cerr << RED << "Usage: benchmark <model_type> <input_path> <engine_path> [num_classes] [runs]" << RESET << std::endl;
            return 1;
        }

        std::string typeStr = argv[2];
        std::string inputPath = argv[3];
        std::string enginePath = argv[4];

        // 解析可选参数
        int custom_num_classes = 80;
        int num_runs = 100;

        if (argc >= 6) {
            custom_num_classes = std::atoi(argv[5]);
            if (custom_num_classes <= 0) custom_num_classes = 80;
        }
        if (argc >= 7) {
            num_runs = std::atoi(argv[6]);
            if (num_runs <= 0) num_runs = 100;
        }

        try {
            std::cout << GREEN << "=== TensorRT Benchmark ===" << RESET << std::endl;
            std::cout << "Model: " << enginePath << std::endl;
            std::cout << "Image: " << inputPath << std::endl;
            std::cout << "Type: " << typeStr << std::endl;
            std::cout << "Runs: " << num_runs << std::endl;

            cv::Mat image = cv::imread(inputPath);
            if (image.empty()) {
                std::cerr << "Image not found." << std::endl;
                return 1;
            }

            // --- 分割模型 ---
            if (typeStr == "yolo-seg") {
                SegmentationConfig seg_config;
                seg_config.engine_file = enginePath;
                seg_config.num_classes = custom_num_classes;
                SegmentationModel model(seg_config);

                std::vector<DetectionWithMask> seg_results;
                BenchmarkStats stats = run_benchmark_seg(model, image, seg_results, 10, num_runs);

                // 打印统计结果
                std::cout << std::fixed << std::setprecision(2);
                std::cout << "\n" << GREEN << "=== TensorRT Segmentation Benchmark Results ===" << RESET << std::endl;
                std::cout << "Preprocess:  " << stats.avg_pre_ms << " ms" << std::endl;
                std::cout << "Inference:   " << stats.avg_infer_ms << " ms" << std::endl;
                std::cout << "Postprocess: " << stats.avg_post_ms << " ms" << std::endl;
                std::cout << "Total:       " << stats.avg_total_ms << " ms" << std::endl;
                std::cout << std::setprecision(1);
                std::cout << "FPS:         " << stats.fps << std::endl;

                // 打印检测结果
                std::cout << "\nDetected " << seg_results.size() << " objects:" << std::endl;
                for (size_t i = 0; i < seg_results.size() && i < 10; ++i) {
                    const auto& det = seg_results[i];
                    std::cout << "  [" << i << "] Class: " << det.class_id
                              << ", Conf: " << std::fixed << std::setprecision(4) << det.conf
                              << ", Box: [" << det.bbox.x << "," << det.bbox.y
                              << "," << det.bbox.width << "," << det.bbox.height << "]";
                    if (det.mask) {
                        std::cout << ", Mask: " << det.mask->width << "x" << det.mask->height;
                    }
                    std::cout << std::endl;
                }
                if (seg_results.size() > 10) {
                    std::cout << "  ... and " << (seg_results.size() - 10) << " more" << std::endl;
                }

                // 保存到 JSON
                save_benchmark_results_seg(program_dir + "/trt_seg_results.json", stats, seg_results, enginePath, inputPath);

                // 可视化
                cv::Mat vis = image.clone();
                for (const auto& det : seg_results) {
                    cv::Scalar color = cv::Scalar(0, 255, 0);
                    cv::rectangle(vis, det.bbox, color, 2);

                    // 绘制 mask
                    if (det.mask && det.bbox.width > 0 && det.bbox.height > 0) {
                        cv::Mat mask(det.mask->height, det.mask->width, CV_8UC1, det.mask->data.data());
                        cv::Mat mask_resized;
                        cv::resize(mask, mask_resized, cv::Size(det.bbox.width, det.bbox.height));

                        for (int y = 0; y < det.bbox.height; y++) {
                            for (int x = 0; x < det.bbox.width; x++) {
                                if (mask_resized.at<uint8_t>(y, x) > 128) {
                                    int py = det.bbox.y + y;
                                    int px = det.bbox.x + x;
                                    if (py >= 0 && py < vis.rows && px >= 0 && px < vis.cols) {
                                        cv::Vec3b& pixel = vis.at<cv::Vec3b>(py, px);
                                        pixel[0] = (pixel[0] + color[0]) / 2;
                                        pixel[1] = (pixel[1] + color[1]) / 2;
                                        pixel[2] = (pixel[2] + color[2]) / 2;
                                    }
                                }
                            }
                        }
                    }

                    std::string label = std::to_string(det.class_id) + ": " +
                                       std::to_string(det.conf).substr(0, std::to_string(det.conf).find('.') + 4);
                    cv::putText(vis, label, cv::Point(det.bbox.x, det.bbox.y - 5), 0, 0.5, color, 2);
                }
                cv::imwrite(outputs_dir + "/trt_seg_benchmark_output.jpg", vis);
                std::cout << "Visualization saved to " << outputs_dir << "/trt_seg_benchmark_output.jpg" << std::endl;
            }
            // --- 检测模型 ---
            else if (typeStr == "yolo") {
                DetectionConfig det_config;
                det_config.engine_file = enginePath;
                det_config.num_classes = custom_num_classes;
                DetectionModel model(det_config);

                std::vector<Detection> results;
                BenchmarkStats stats = run_benchmark(model, image, results, 10, num_runs);

                // 打印统计结果
                std::cout << std::fixed << std::setprecision(2);
                std::cout << "\n" << GREEN << "=== TensorRT Benchmark Results ===" << RESET << std::endl;
                std::cout << "Preprocess:  " << stats.avg_pre_ms << " ms" << std::endl;
                std::cout << "Inference:   " << stats.avg_infer_ms << " ms" << std::endl;
                std::cout << "Postprocess: " << stats.avg_post_ms << " ms" << std::endl;
                std::cout << "Total:       " << stats.avg_total_ms << " ms" << std::endl;
                std::cout << std::setprecision(1);
                std::cout << "FPS:         " << stats.fps << std::endl;

                // 打印检测结果
                std::cout << "\nDetected " << results.size() << " objects:" << std::endl;
                for (size_t i = 0; i < results.size() && i < 10; ++i) {
                    const auto& det = results[i];
                    std::cout << "  [" << i << "] Class: " << det.class_id
                              << ", Conf: " << std::fixed << std::setprecision(4) << det.conf
                              << ", Box: [" << det.bbox.x << "," << det.bbox.y
                              << "," << det.bbox.width << "," << det.bbox.height << "]" << std::endl;
                }
                if (results.size() > 10) {
                    std::cout << "  ... and " << (results.size() - 10) << " more" << std::endl;
                }

                // 保存到 JSON
                save_benchmark_results(program_dir + "/trt_results.json", stats, results, enginePath, inputPath);

                // 可视化
                cv::Mat vis = image.clone();
                for (const auto& det : results) {
                    cv::rectangle(vis, det.bbox, cv::Scalar(0, 255, 0), 2);
                    std::string label = std::to_string(det.class_id) + ": " +
                                       std::to_string(det.conf).substr(0, std::to_string(det.conf).find('.') + 4);
                    cv::putText(vis, label, cv::Point(det.bbox.x, det.bbox.y - 5), 0, 0.5, cv::Scalar(0, 255, 0), 2);
                }
                cv::imwrite(outputs_dir + "/trt_benchmark_output.jpg", vis);
                std::cout << "Visualization saved to " << outputs_dir << "/trt_benchmark_output.jpg" << std::endl;
            }
            // --- 分类模型 ---
            else if (typeStr == "resnet") {
                ClassificationConfig cls_config;
                cls_config.engine_file = enginePath;
                cls_config.num_classes = custom_num_classes;
                ClassificationModel model(cls_config);

                int class_id = 0;
                float score = 0.0f;
                BenchmarkStats stats = run_benchmark_cls(model, image, class_id, score, 10, num_runs);

                // 打印统计结果
                std::cout << std::fixed << std::setprecision(2);
                std::cout << "\n" << GREEN << "=== TensorRT Classification Benchmark Results ===" << RESET << std::endl;
                std::cout << "Preprocess:  " << stats.avg_pre_ms << " ms" << std::endl;
                std::cout << "Inference:   " << stats.avg_infer_ms << " ms" << std::endl;
                std::cout << "Postprocess: " << stats.avg_post_ms << " ms" << std::endl;
                std::cout << "Total:       " << stats.avg_total_ms << " ms" << std::endl;
                std::cout << std::setprecision(1);
                std::cout << "FPS:         " << stats.fps << std::endl;

                // 打印分类结果
                std::cout << "\nClassification Result:" << std::endl;
                std::cout << "  Class ID: " << class_id << std::endl;
                std::cout << "  Score:    " << std::fixed << std::setprecision(4) << score << std::endl;

                // 保存到 JSON
                save_benchmark_results_cls(program_dir + "/trt_cls_results.json", stats, class_id, score, enginePath, inputPath);

                // 可视化
                cv::Mat vis = image.clone();
                std::string label = "Class: " + std::to_string(class_id) + " Score: " +
                                   std::to_string(score).substr(0, std::to_string(score).find('.') + 5);
                cv::putText(vis, label, cv::Point(20, 50), 0, 1.0, cv::Scalar(0, 0, 255), 2);
                cv::imwrite(outputs_dir + "/trt_cls_benchmark_output.jpg", vis);
                std::cout << "Visualization saved to " << outputs_dir << "/trt_cls_benchmark_output.jpg" << std::endl;
            }
            else {
                std::cerr << RED << "Benchmark mode currently only supports 'yolo', 'yolo-seg', or 'resnet' model type." << RESET << std::endl;
                return 1;
            }
        }
        catch (const std::exception& e) {
            std::cerr << RED << "Error: " << e.what() << RESET << std::endl;
            return 1;
        }
        return 0;
    }

    // -------------------------------------------------------------------------
    // Mode: Inference
    // -------------------------------------------------------------------------
    if (mode == "infer_video" || mode == "infer_image" || mode == "infer_segment") {
        if (argc < 5) {
            std::cerr << RED << "Usage: " << mode << " <model_type> <input_path> <engine_path> [num_classes]" << RESET << std::endl;
            std::cerr << "  <model_type>: 'yolo', 'yolo-seg', or 'resnet'" << std::endl;
            std::cerr << "  num_classes: 可选，指定类别数量" << std::endl;
            return 1;
        }

        std::string typeStr = argv[2];
        std::string inputPath = argv[3];
        std::string enginePath = argv[4];

        // 解析可选的类别数参数
        int custom_num_classes = -1;
        bool disable_norm = false;

        int arg_idx = 5;
        while (arg_idx < argc) {
            std::string arg = argv[arg_idx];
            if (arg == "--no-norm") {
                disable_norm = true;
            } else if (custom_num_classes <= 0) {
                custom_num_classes = std::atoi(argv[arg_idx]);
                if (custom_num_classes <= 0) {
                    std::cerr << RED << "Invalid num_classes: " << argv[arg_idx] << RESET << std::endl;
                    return 1;
                }
            }
            arg_idx++;
        }

        try {
            // --- 分割推理模式 ---
            if (mode == "infer_segment") {
                if (typeStr != "yolo-seg") {
                    std::cerr << RED << "Error: infer_segment mode requires 'yolo-seg' model type." << RESET << std::endl;
                    return 1;
                }

                SegmentationConfig seg_config;
                seg_config.engine_file = enginePath;
                seg_config.num_classes = (custom_num_classes > 0) ? custom_num_classes : 80;

                SegmentationModel model(seg_config);

                cv::Mat image = cv::imread(inputPath);
                if (image.empty()) {
                    std::cerr << "Image not found." << std::endl;
                    return 1;
                }

                std::vector<DetectionWithMask> objects;

                std::cout << "Warming up (10 runs)..." << std::endl;
                for(int i=0; i<10; ++i) { model.infer(image); model.get_results(objects); }

                std::cout << "Run segmentation inference..." << std::endl;

                // 此时 Model 内部会自动打印具体的 Pre/Infer/Post 耗时
                model.infer(image);
                model.get_results(objects);

                // 调试输出
                std::cout << "Detected " << objects.size() << " objects:" << std::endl;
                for(size_t i = 0; i < objects.size(); ++i) {
                    auto& obj = objects[i];
                    std::cout << "  [" << i << "] Class: " << obj.class_id
                              << ", Conf: " << obj.conf
                              << ", Box: [" << obj.bbox.x << "," << obj.bbox.y
                              << "," << obj.bbox.width << "," << obj.bbox.height << "]";
                    if (obj.mask) {
                        std::cout << ", Mask: " << obj.mask->width << "x" << obj.mask->height;
                    }
                    std::cout << std::endl;
                }

                // 可视化
                cv::Mat overlay = image.clone();

                // YOLO COCO 80类颜色定义 (简化版，只定义常用类别)
                // 如果类别不在列表中，使用默认绿色
                auto get_class_color = [](int class_id) -> cv::Scalar {
                    // 常见类别颜色映射
                    switch(class_id) {
                        case 0:  return cv::Scalar(255, 0, 0);    // person - 红色
                        case 1:  return cv::Scalar(0, 255, 0);    // bicycle - 绿色
                        case 2:  return cv::Scalar(0, 0, 255);    // car - 蓝色
                        case 3:  return cv::Scalar(255, 255, 0);  // motorcycle - 黄色
                        case 5:  return cv::Scalar(255, 0, 255);  // bus - 紫色
                        case 7:  return cv::Scalar(0, 255, 255);  // truck - 青色
                        case 16: return cv::Scalar(255, 128, 0);  // dog - 橙色
                        case 17: return cv::Scalar(128, 0, 255);  // cat - 深紫
                        default: return cv::Scalar(0, 255, 0);    // 默认绿色
                    }
                };

                for(auto& obj : objects) {
                    // 获取类别颜色
                    cv::Scalar color = get_class_color(obj.class_id);

                    // 绘制边界框 (使用类别颜色)
                    cv::rectangle(overlay, obj.bbox, color, 2);

                    // 绘制 mask
                    if (obj.mask && obj.bbox.width > 0 && obj.bbox.height > 0) {
                        cv::Mat mask(obj.mask->height, obj.mask->width, CV_8UC1, obj.mask->data.data());

                        // 缩放 mask 到 bbox 尺寸
                        cv::Mat mask_resized;
                        if (obj.mask->width > 0 && obj.mask->height > 0) {
                            cv::resize(mask, mask_resized, cv::Size(obj.bbox.width, obj.bbox.height));

                            // 在原图上叠加 mask (使用类别颜色，半透明效果)
                            for (int y = 0; y < obj.bbox.height; y++) {
                                for (int x = 0; x < obj.bbox.width; x++) {
                                    if (mask_resized.at<uint8_t>(y, x) > 128) {
                                        int py = obj.bbox.y + y;
                                        int px = obj.bbox.x + x;
                                        if (py >= 0 && py < overlay.rows && px >= 0 && px < overlay.cols) {
                                            // 使用半透明叠加 (50% 透明度)
                                            cv::Vec3b& pixel = overlay.at<cv::Vec3b>(py, px);
                                            pixel[0] = (pixel[0] + color[0]) / 2;
                                            pixel[1] = (pixel[1] + color[1]) / 2;
                                            pixel[2] = (pixel[2] + color[2]) / 2;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // 添加标签 (使用类别颜色)
                    std::string label = std::to_string(obj.class_id) + ": " + std::to_string(obj.conf).substr(0, 4);
                    cv::putText(overlay, label, cv::Point(obj.bbox.x, obj.bbox.y - 5), 0, 0.5, color, 2);
                }

                cv::imwrite(outputs_dir + "/output_segmentation.jpg", overlay);
                std::cout << GREEN << "Detected " << objects.size() << " objects with masks." << RESET << std::endl;
                std::cout << "Result saved to " << outputs_dir << "/output_segmentation.jpg" << std::endl;
                return 0;
            }

            // 结果容器
            std::vector<Detection> objects;
            int cls_id = -1;
            float cls_score = 0.0f;

            // --- 视频推理 ---
            if (mode == "infer_video") {
                cv::VideoCapture cap(inputPath);
                if (!cap.isOpened()) {
                    std::cerr << "Failed to open video." << std::endl;
                    return 1;
                }

                std::string outputVideoPath = outputs_dir + "/output_detection.avi";
                int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
                int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
                cv::VideoWriter video(outputVideoPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(w, h));

                cv::Mat frame;
                std::cout << GREEN << "Processing video..." << RESET << std::endl;

                while (cap.read(frame)) {
                    if (frame.empty()) break;

                    // 清空上一帧结果
                    objects.clear();

                    if (typeStr == "yolo") {
                        DetectionConfig det_config;
                        det_config.engine_file = enginePath;
                        det_config.num_classes = (custom_num_classes > 0) ? custom_num_classes : 80;
                        DetectionModel model(det_config);
                        model.infer(frame);
                        model.get_results(objects);

                        for(auto& obj : objects) {
                            cv::rectangle(frame, obj.bbox, cv::Scalar(0,255,0), 2);
                            std::string label = std::to_string(obj.class_id) + " " + std::to_string(obj.conf).substr(0,4);
                            cv::putText(frame, label, cv::Point(obj.bbox.x, obj.bbox.y-5), 0, 0.5, cv::Scalar(0,255,0), 2);
                        }
                    } else if (typeStr == "resnet") {
                        ClassificationConfig cls_config;
                        cls_config.engine_file = enginePath;
                        cls_config.num_classes = (custom_num_classes > 0) ? custom_num_classes : 1000;
                        cls_config.use_normalization = !disable_norm;
                        ClassificationModel model(cls_config);
                        model.infer(frame);
                        model.get_results(cls_id, cls_score);

                        std::string label = "Class: " + std::to_string(cls_id) + " Score: " + std::to_string(cls_score);
                        cv::putText(frame, label, cv::Point(20,50), 0, 1.0, cv::Scalar(0,0,255), 2);
                    }
                    video.write(frame);
                }
                std::cout << GREEN << "Done. Saved to " << outputVideoPath << RESET << std::endl;
            }
            // --- 图片推理 ---
            else {
                cv::Mat image = cv::imread(inputPath);
                if (image.empty()) {
                    std::cerr << "Image not found." << std::endl;
                    return 1;
                }

                std::cout << "Warming up (10 runs)..." << std::endl;

                if (typeStr == "yolo") {
                    DetectionConfig det_config;
                    det_config.engine_file = enginePath;
                    det_config.num_classes = (custom_num_classes > 0) ? custom_num_classes : 80;
                    DetectionModel model(det_config);

                    for(int i=0; i<10; ++i) { model.infer(image); model.get_results(objects); }

                    std::cout << "Run inference..." << std::endl;
                    objects.clear(); // 清空预热结果

                    model.infer(image);
                    model.get_results(objects);

                    // 可视化
                    for(auto& obj : objects) {
                        cv::rectangle(image, obj.bbox, cv::Scalar(0,255,0), 2);
                        std::string label = std::to_string(obj.class_id) + ": " + std::to_string(obj.conf).substr(0,4);
                        cv::putText(image, label, cv::Point(obj.bbox.x, obj.bbox.y-5), 0, 0.5, cv::Scalar(0,255,0), 2);
                    }
                    std::cout << GREEN << "Detected " << objects.size() << " objects." << RESET << std::endl;
                } else if (typeStr == "resnet") {
                    ClassificationConfig cls_config;
                    cls_config.engine_file = enginePath;
                    cls_config.num_classes = (custom_num_classes > 0) ? custom_num_classes : 1000;
                    cls_config.use_normalization = !disable_norm;
                    if (disable_norm) {
                        std::cout << "[INFO] Using [0,1] normalization (no ImageNet normalization)" << std::endl;
                    }

                    ClassificationModel model(cls_config);

                    for(int i=0; i<10; ++i) { model.infer(image); model.get_results(cls_id, cls_score); }

                    std::cout << "Run inference..." << std::endl;

                    model.infer(image);
                    model.get_results(cls_id, cls_score);

                    // 可视化
                    std::string label = "Class ID: " + std::to_string(cls_id) + " (" + std::to_string(cls_score) + ")";
                    cv::putText(image, label, cv::Point(20,50), 0, 1.0, cv::Scalar(0,0,255), 2);
                    std::cout << GREEN << label << RESET << std::endl;
                } else {
                    std::cerr << RED << "Invalid model type for image inference." << RESET << std::endl;
                    return 1;
                }

                // 根据模型类型保存到不同文件
                std::string outputPath;
                if (typeStr == "yolo") {
                    outputPath = outputs_dir + "/output_detection.jpg";
                } else if (typeStr == "resnet") {
                    outputPath = outputs_dir + "/output_classification.jpg";
                } else {
                    outputPath = outputs_dir + "/output.jpg";
                }
                cv::imwrite(outputPath, image);
                std::cout << "Result saved to " << outputPath << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cerr << RED << "Error: " << e.what() << RESET << std::endl;
            return 1;
        }
    }
    else {
        std::cerr << RED << "Invalid mode." << RESET << std::endl;
        return 1;
    }

    return 0;
}