#include "preprocess.h"
#include "cuda_utils.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp> // 需要 OpenCV 进行矩阵求逆

// Host and device pointers for image buffers
static uint8_t* img_buffer_host = nullptr;    // Pinned memory on the host for faster transfers
static uint8_t* img_buffer_device = nullptr;  // Memory on the device (GPU)

// Structure to represent a 2x3 affine transformation matrix
struct AffineMatrix {
    float value[6]; // [m00, m01, m02, m10, m11, m12]
};

// =========================================================================
// Kernel: YOLO 专用预处理 (Letterbox + BGR2RGB + Normalize[0,1])
// 优化点：
// - 硬编码 mean=0, std=1 (YOLO 标准)
// - 硬编码填充色 114
// - 减少参数传递
// =========================================================================
__global__ void yolo_preprocess_kernel(
    uint8_t* src,           // Source image on device (BGR)
    int src_line_size,      // src_width * 3
    int src_width,          // Source image width
    int src_height,         // Source image height
    float* dst,             // Destination NCHW buffer (RGB, normalized)
    int dst_width,          // Target width (e.g., 640)
    int dst_height,         // Target height (e.g., 640)
    AffineMatrix d2s,       // Affine matrix (dst -> src)
    int edge                // Total pixels = dst_width * dst_height
) {
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    // 提取仿射矩阵
    float m_x1 = d2s.value[0];
    float m_y1 = d2s.value[1];
    float m_z1 = d2s.value[2];
    float m_x2 = d2s.value[3];
    float m_y2 = d2s.value[4];
    float m_z2 = d2s.value[5];

    // 目标坐标
    int dx = position % dst_width;
    int dy = position / dst_width;

    // 映射到源图像
    float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
    float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;

    float r, g, b;

    // 边界检查 - YOLO 填充 114
    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
        r = g = b = 114.0f;
    }
    else {
        // 双线性插值
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;

        float w1 = hy * hx; // Top-left
        float w2 = hy * lx; // Top-right
        float w3 = ly * hx; // Bottom-left
        float w4 = ly * lx; // Bottom-right

        // 边界安全的像素读取
        uint8_t p1[3] = {114, 114, 114};
        uint8_t p2[3] = {114, 114, 114};
        uint8_t p3[3] = {114, 114, 114};
        uint8_t p4[3] = {114, 114, 114};

        if (y_low >= 0) {
            if (x_low >= 0) {
                uint8_t* ptr = src + y_low * src_line_size + x_low * 3;
                p1[0] = ptr[0]; p1[1] = ptr[1]; p1[2] = ptr[2];
            }
            if (x_high < src_width) {
                uint8_t* ptr = src + y_low * src_line_size + x_high * 3;
                p2[0] = ptr[0]; p2[1] = ptr[1]; p2[2] = ptr[2];
            }
        }

        if (y_high < src_height) {
            if (x_low >= 0) {
                uint8_t* ptr = src + y_high * src_line_size + x_low * 3;
                p3[0] = ptr[0]; p3[1] = ptr[1]; p3[2] = ptr[2];
            }
            if (x_high < src_width) {
                uint8_t* ptr = src + y_high * src_line_size + x_high * 3;
                p4[0] = ptr[0]; p4[1] = ptr[1]; p4[2] = ptr[2];
            }
        }

        // BGR 插值
        b = w1 * p1[0] + w2 * p2[0] + w3 * p3[0] + w4 * p4[0];
        g = w1 * p1[1] + w2 * p2[1] + w3 * p3[1] + w4 * p4[1];
        r = w1 * p1[2] + w2 * p2[2] + w3 * p3[2] + w4 * p4[2];
    }

    // BGR → RGB (直接交换)
    float temp = r;
    r = b;
    b = temp;

    // 归一化: x / 255.0 (YOLO 标准: mean=0, std=1)
    r = r / 255.0f;
    g = g / 255.0f;
    b = b / 255.0f;

    // NCHW 布局输出
    int area = dst_width * dst_height;
    dst[dy * dst_width + dx] = r;                    // R channel
    dst[area + dy * dst_width + dx] = g;             // G channel
    dst[2 * area + dy * dst_width + dx] = b;         // B channel
}

// =========================================================================
// Kernel: ResNet 专用预处理 (Stretch + BGR2RGB + ImageNet Normalize)
// 优化点：
// - 支持自定义 mean/std
// - 拉伸模式 (忽略宽高比)
// - 填充色 0
// =========================================================================
__global__ void resnet_preprocess_kernel(
    uint8_t* src,           // Source image on device (BGR)
    int src_line_size,      // src_width * 3
    int src_width,          // Source image width
    int src_height,         // Source image height
    float* dst,             // Destination NCHW buffer (RGB, normalized)
    int dst_width,          // Target width (e.g., 224)
    int dst_height,         // Target height (e.g., 224)
    float scale_x,          // dst_width / src_width
    float scale_y,          // dst_height / src_height
    float mean_r,           // ImageNet mean R
    float mean_g,           // ImageNet mean G
    float mean_b,           // ImageNet mean B
    float std_r,            // ImageNet std R
    float std_g,            // ImageNet std G
    float std_b,            // ImageNet std B
    int edge                // Total pixels = dst_width * dst_height
) {
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    // 目标坐标
    int dx = position % dst_width;
    int dy = position / dst_width;

    // 映射到源图像 (拉伸模式，无偏移)
    float src_x = dx / scale_x;
    float src_y = dy / scale_y;

    float r, g, b;

    // 边界检查 - ResNet 填充 0
    if (src_x < 0 || src_x >= src_width || src_y < 0 || src_y >= src_height) {
        r = g = b = 0.0f;
    }
    else {
        // 双线性插值
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = min(y_low + 1, src_height - 1);
        int x_high = min(x_low + 1, src_width - 1);

        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;

        float w1 = hy * hx;
        float w2 = hy * lx;
        float w3 = ly * hx;
        float w4 = ly * lx;

        // 读取 4 个像素 (已保证边界安全)
        uint8_t* p1 = src + y_low * src_line_size + x_low * 3;
        uint8_t* p2 = src + y_low * src_line_size + x_high * 3;
        uint8_t* p3 = src + y_high * src_line_size + x_low * 3;
        uint8_t* p4 = src + y_high * src_line_size + x_high * 3;

        // BGR 插值
        b = w1 * p1[0] + w2 * p2[0] + w3 * p3[0] + w4 * p4[0];
        g = w1 * p1[1] + w2 * p2[1] + w3 * p3[1] + w4 * p4[1];
        r = w1 * p1[2] + w2 * p2[2] + w3 * p3[2] + w4 * p4[2];
    }

    // ImageNet 归一化 + BGR → RGB
    // 输入是 BGR，输出是 RGB (NCHW 顺序)
    // ImageNet mean (RGB): R=0.485, G=0.456, B=0.406
    // ImageNet std (RGB):  R=0.229, G=0.224, B=0.225
    //
    // 正确的转换:
    //   R_final = (r / 255.0 - mean_r) / std_r   # 原始 R 值用 R 的参数
    //   G_final = (g / 255.0 - mean_g) / std_g   # 原始 G 值用 G 的参数
    //   B_final = (b / 255.0 - mean_b) / std_b   # 原始 B 值用 B 的参数
    float r_final = (r / 255.0f - mean_r) / std_r;   // R → R
    float g_final = (g / 255.0f - mean_g) / std_g;   // G → G
    float b_final = (b / 255.0f - mean_b) / std_b;   // B → B

    // NCHW 布局输出
    int area = dst_width * dst_height;
    dst[dy * dst_width + dx] = r_final;
    dst[area + dy * dst_width + dx] = g_final;
    dst[2 * area + dy * dst_width + dx] = b_final;
}

// =========================================================================
// Kernel: ImageNet 标准 CenterCrop 预处理 (使用 Bilinear 插值)
// 流程:
// 1. 先将图像缩放到较小边等于目标尺寸 (保持宽高比)
// 2. 然后从中心裁剪到目标尺寸
// 3. BGR2RGB + ImageNet 归一化
// 注意: PIL 使用的是特殊的 Bicubic 算法，难以在 CUDA 中完全复现
//      这里使用 Bilinear 插值作为折中方案
// =========================================================================
__global__ void center_crop_preprocess_kernel(
    uint8_t* src,           // Source image on device (BGR)
    int src_line_size,      // src_width * 3
    int src_width,          // Source image width
    int src_height,         // Source image height
    float* dst,             // Destination NCHW buffer (RGB, normalized)
    int dst_width,          // Target width (e.g., 224)
    int dst_height,         // Target height (e.g., 224)
    int resize_width,       // Intermediate resize width
    int resize_height,      // Intermediate resize height
    int crop_x_offset,      // X offset for center crop
    int crop_y_offset,      // Y offset for center crop
    float mean_r,           // ImageNet mean R
    float mean_g,           // ImageNet mean G
    float mean_b,           // ImageNet mean B
    float std_r,            // ImageNet std R
    float std_g,            // ImageNet std G
    float std_b,            // ImageNet std B
    int edge                // Total pixels = dst_width * dst_height
) {
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    // 目标坐标 (在裁剪后的图像中)
    int dx = position % dst_width;
    int dy = position / dst_width;

    // 映射到缩放后的图像坐标 (加上裁剪偏移)
    float resize_x = dx + crop_x_offset;
    float resize_y = dy + crop_y_offset;

    // 映射到源图像坐标
    float scale_x = (float)src_width / resize_width;
    float scale_y = (float)src_height / resize_height;
    float src_x = resize_x * scale_x;
    float src_y = resize_y * scale_y;

    float r, g, b;

    // 边界检查 - ImageNet 填充 0
    if (src_x < 0 || src_x >= src_width || src_y < 0 || src_y >= src_height) {
        r = g = b = 0.0f;
    }
    else {
        // 双线性插值
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = min(y_low + 1, src_height - 1);
        int x_high = min(x_low + 1, src_width - 1);

        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;

        float w1 = hy * hx;
        float w2 = hy * lx;
        float w3 = ly * hx;
        float w4 = ly * lx;

        // 读取 4 个像素 (已保证边界安全)
        uint8_t* p1 = src + y_low * src_line_size + x_low * 3;
        uint8_t* p2 = src + y_low * src_line_size + x_high * 3;
        uint8_t* p3 = src + y_high * src_line_size + x_low * 3;
        uint8_t* p4 = src + y_high * src_line_size + x_high * 3;

        // BGR 插值
        b = w1 * p1[0] + w2 * p2[0] + w3 * p3[0] + w4 * p4[0];
        g = w1 * p1[1] + w2 * p2[1] + w3 * p3[1] + w4 * p4[1];
        r = w1 * p1[2] + w2 * p2[2] + w3 * p3[2] + w4 * p4[2];
    }

    // ImageNet 归一化 + BGR → RGB
    // 输入是 BGR，输出是 RGB (NCHW)
    // 输入: b, g, r (从 BGR 图像读取)
    // 输出: R_final, G_final, B_final (归一化后的 RGB 值)
    //
    // ImageNet mean (RGB): R=0.485, G=0.456, B=0.406
    // ImageNet std (RGB):  R=0.229, G=0.224, B=0.225
    //
    // 正确的转换:
    //   R_final = (r / 255.0 - mean_r) / std_r   # 原始 R 值用 R 的参数
    //   G_final = (g / 255.0 - mean_g) / std_g   # 原始 G 值用 G 的参数
    //   B_final = (b / 255.0 - mean_b) / std_b   # 原始 B 值用 B 的参数
    float r_final = (r / 255.0f - mean_r) / std_r;   // R → R
    float g_final = (g / 255.0f - mean_g) / std_g;   // G → G
    float b_final = (b / 255.0f - mean_b) / std_b;   // B → B

    // NCHW 布局输出
    int area = dst_width * dst_height;
    dst[dy * dst_width + dx] = r_final;
    dst[area + dy * dst_width + dx] = g_final;
    dst[2 * area + dy * dst_width + dx] = b_final;
}

// =========================================================================
// Host Function - 根据模式调用专用核函数
// =========================================================================
void cuda_preprocess(
    uint8_t* src,        // Source image data on host
    int src_width,       // Source image width
    int src_height,      // Source image height
    float* dst,          // Destination buffer on device
    int dst_width,       // Destination image width
    int dst_height,      // Destination image height
    cudaStream_t stream,  // CUDA stream
    const float* mean,   // Mean (for ResNet)
    const float* std,    // Std (for ResNet)
    PreprocessMode mode  // Scaling mode
) {
    // 1. 内存拷贝 Host -> Device
    int img_size = src_width * src_height * 3;
    memcpy(img_buffer_host, src, img_size);

    CUDA_CHECK(cudaMemcpyAsync(
        img_buffer_device,
        img_buffer_host,
        img_size,
        cudaMemcpyHostToDevice,
        stream
    ));

    // 2. 计算并行配置
    int jobs = dst_height * dst_width;
    int threads = 256;
    int blocks = (jobs + threads - 1) / threads;

    // 3. 根据模式调用专用核函数
    if (mode == MODE_LETTERBOX) {
        // ==================== YOLO 模式 ====================
        // 计算 Letterbox 仿射变换矩阵
        AffineMatrix s2d, d2s;
        float scale = std::min(
            dst_height / (float)src_height,
            dst_width / (float)src_width
        );

        s2d.value[0] = scale;
        s2d.value[1] = 0;
        s2d.value[2] = -scale * src_width * 0.5f + dst_width * 0.5f;
        s2d.value[3] = 0;
        s2d.value[4] = scale;
        s2d.value[5] = -scale * src_height * 0.5f + dst_height * 0.5f;

        // 计算逆矩阵
        cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
        cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
        cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);
        memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

        // 调用 YOLO 专用核函数
        yolo_preprocess_kernel<<<blocks, threads, 0, stream>>>(
            img_buffer_device,
            src_width * 3,
            src_width,
            src_height,
            dst,
            dst_width,
            dst_height,
            d2s,
            jobs
        );
    }
    else if (mode == MODE_CENTER_CROP) {
        // ==================== ImageNet CenterCrop 模式 ====================
        // PyTorch 标准预处理:
        // 1. transforms.Resize(256) - 将较短边缩放到 256
        // 2. transforms.CenterCrop(224) - 从中心裁剪出 224x224

        int resize_width, resize_height;
        const int resize_size = 256;  // ImageNet 标准尺寸

        // 计算缩放后的尺寸 (保持宽高比，使较小边等于 256)
        if (src_width < src_height) {
            // 宽度是较小边
            resize_width = resize_size;
            resize_height = (int)round((float)src_height * resize_size / src_width);
        } else {
            // 高度是较小边
            resize_height = resize_size;
            resize_width = (int)round((float)src_width * resize_size / src_height);
        }

        // 计算中心裁剪的偏移量 (从 256x? 或 ?x256 裁剪到 224x224)
        int crop_x_offset = (resize_width - dst_width) / 2;
        int crop_y_offset = (resize_height - dst_height) / 2;

        // 使用传入的 mean/std，如果没有则使用 ImageNet 默认值
        float default_mean[3] = {0.485f, 0.456f, 0.406f};
        float default_std[3]  = {0.229f, 0.224f, 0.225f};

        const float* use_mean = mean ? mean : default_mean;
        const float* use_std  = std  ? std  : default_std;

        // 调用 CenterCrop 专用核函数
        center_crop_preprocess_kernel<<<blocks, threads, 0, stream>>>(
            img_buffer_device,
            src_width * 3,
            src_width,
            src_height,
            dst,
            dst_width,
            dst_height,
            resize_width,
            resize_height,
            crop_x_offset,
            crop_y_offset,
            use_mean[0], use_mean[1], use_mean[2],
            use_std[0], use_std[1], use_std[2],
            jobs
        );
    }
    else {
        // ==================== ResNet STRETCH 模式 ====================
        // 计算 Stretch 缩放因子
        float scale_x = (float)dst_width / src_width;
        float scale_y = (float)dst_height / src_height;

        // 使用传入的 mean/std，如果没有则使用 ImageNet 默认值
        float default_mean[3] = {0.485f, 0.456f, 0.406f};
        float default_std[3]  = {0.229f, 0.224f, 0.225f};

        const float* use_mean = mean ? mean : default_mean;
        const float* use_std  = std  ? std  : default_std;

        // 调用 ResNet 专用核函数
        resnet_preprocess_kernel<<<blocks, threads, 0, stream>>>(
            img_buffer_device,
            src_width * 3,
            src_width,
            src_height,
            dst,
            dst_width,
            dst_height,
            scale_x,
            scale_y,
            use_mean[0], use_mean[1], use_mean[2],
            use_std[0], use_std[1], use_std[2],
            jobs
        );
    }

    CUDA_CHECK(cudaGetLastError());
}

void cuda_preprocess_init(int max_image_size) {
    CUDA_CHECK(cudaMallocHost((void**)&img_buffer_host, max_image_size * 3));
    CUDA_CHECK(cudaMalloc((void**)&img_buffer_device, max_image_size * 3));
}

void cuda_preprocess_destroy() {
    CUDA_CHECK(cudaFree(img_buffer_device));
    CUDA_CHECK(cudaFreeHost(img_buffer_host));
}

// =========================================================================
// Kernel: 解码单个 mask (mask_coeff @ mask_proto)
// =========================================================================
__global__ void decode_single_mask_kernel(
    float* mask_coeff,      // [N, 32] - 每个检测框的系数
    float* mask_proto,      // [1, 32, H, W] - mask 原型
    unsigned char* mask_out, // [N, H, W] - 输出 mask
    int num_detections,     // 检测框数量
    int mask_dim,           // 系数维度 (32)
    int proto_h,            // 原型高度
    int proto_w,            // 原型宽度
    int out_h,              // 输出高度
    int out_w               // 输出宽度
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_detections) return;

    // 每个线程处理一个检测框的 mask
    float* coeff = mask_coeff + idx * mask_dim;
    unsigned char* out = mask_out + idx * out_h * out_w;

    // 遍历每个像素位置
    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            float sum = 0.0f;

            // mask_coeff @ mask_proto[:, y, x]
            for (int c = 0; c < mask_dim; c++) {
                float proto_val = mask_proto[c * proto_h * proto_w + y * proto_w + x];
                sum += coeff[c] * proto_val;
            }

            // Sigmoid 激活
            float alpha = 1.0f / (1.0f + expf(-sum));

            // 二值化 (0 或 255)
            out[y * out_w + x] = (alpha > 0.5f) ? 255 : 0;
        }
    }
}

// Host 函数：调用 mask 解码 Kernel
void cuda_decode_masks(
    float* mask_coeff_device,    // GPU: [N, mask_dim]
    float* mask_proto_device,    // GPU: [1, mask_dim, proto_h, proto_w]
    unsigned char* mask_out_device, // GPU: [N, out_h, out_w]
    int num_detections,
    int mask_dim,
    int proto_h,
    int proto_w,
    int out_h,
    int out_w,
    cudaStream_t stream
) {
    if (num_detections == 0) return;

    int threads = 256;
    int blocks = (num_detections + threads - 1) / threads;

    decode_single_mask_kernel<<<blocks, threads, 0, stream>>>(
        mask_coeff_device,
        mask_proto_device,
        mask_out_device,
        num_detections,
        mask_dim,
        proto_h,
        proto_w,
        out_h,
        out_w
    );

    CUDA_CHECK(cudaGetLastError());
}