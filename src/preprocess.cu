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
// Kernel: 仿射变换 + 归一化 (支持 Mean/Std)
// =========================================================================
__global__ void warpaffine_kernel(
    uint8_t* src,           // Source image on device
    int src_line_size,      // Number of bytes per source image row
    int src_width,          // Source image width
    int src_height,         // Source image height
    float* dst,             // Destination image on device (output)
    int dst_width,          // Destination image width
    int dst_height,         // Destination image height
    uint8_t const_value_st, // Constant value for out-of-bound pixels
    AffineMatrix d2s,       // Affine transformation matrix (destination to source)
    int edge,               // Total number of pixels to process
    float mean_0, float mean_1, float mean_2, // Normalization Mean (R, G, B)
    float std_0, float std_1, float std_2     // Normalization Std (R, G, B)
) {
    // Calculate the global position of the thread
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return; // Exit if position exceeds total pixels

    // Extract affine matrix elements
    float m_x1 = d2s.value[0];
    float m_y1 = d2s.value[1];
    float m_z1 = d2s.value[2];
    float m_x2 = d2s.value[3];
    float m_y2 = d2s.value[4];
    float m_z2 = d2s.value[5];

    // Calculate destination pixel coordinates
    int dx = position % dst_width;
    int dy = position / dst_width;

    // Apply affine transformation to get source coordinates
    float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
    float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;

    float c0, c1, c2; // Color channels (will be B, G, R initially)

    // Check if the source coordinates are out of bounds
    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
        // Assign constant value if out of range
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    }
    else {
        // Perform bilinear interpolation
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_value[] = { const_value_st, const_value_st, const_value_st };

        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;

        float w1 = hy * hx; // Top-left
        float w2 = hy * lx; // Top-right
        float w3 = ly * hx; // Bottom-left
        float w4 = ly * lx; // Bottom-right

        uint8_t* v1 = const_value;
        uint8_t* v2 = const_value;
        uint8_t* v3 = const_value;
        uint8_t* v4 = const_value;

        if (y_low >= 0) {
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3;
            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3;
        }

        if (y_high < src_height) {
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;
            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }

        // src is BGR
        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0]; // Blue
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1]; // Green
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2]; // Red
    }

    // Convert from BGR to RGB by swapping channels
    // Before: c0=B, c1=G, c2=R
    float t = c2;
    c2 = c0;
    c0 = t;
    // After: c0=R, c1=G, c2=B

    // Normalize: (x / 255.0 - mean) / std
    // Note: c0 is Red, so use mean_0/std_0
    c0 = (c0 / 255.0f - mean_0) / std_0;
    c1 = (c1 / 255.0f - mean_1) / std_1;
    c2 = (c2 / 255.0f - mean_2) / std_2;

    // Rearrange to Planar format (NCHW)
    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;        // R Plane
    float* pdst_c1 = pdst_c0 + area;                   // G Plane
    float* pdst_c2 = pdst_c1 + area;                   // B Plane

    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}

// =========================================================================
// Host Function
// =========================================================================
void cuda_preprocess(
    uint8_t* src,        // Source image data on host
    int src_width,       // Source image width
    int src_height,      // Source image height
    float* dst,          // Destination buffer on device
    int dst_width,       // Destination image width
    int dst_height,      // Destination image height
    cudaStream_t stream,  // CUDA stream
    const float* mean,   // Added: Mean
    const float* std,    // Added: Std
    PreprocessMode mode  // Added: Scaling mode
) {
    // 1. 设置默认 Mean/Std (如果传入为空，默认兼容 YOLO)
    // YOLO: mean=0, std=1 -> result = x / 255.0
    float default_mean[3] = {0.0f, 0.0f, 0.0f};
    float default_std[3]  = {1.0f, 1.0f, 1.0f};
    
    // 如果是 ResNet 且未传参，通常应该是 ImageNet 均值，但这里为了安全使用默认值
    // 用户调用时应手动传入 {0.485, ...}
    
    const float* use_mean = mean ? mean : default_mean;
    const float* use_std  = std  ? std  : default_std;

    // 2. 内存拷贝 Host -> Device
    int img_size = src_width * src_height * 3;
    memcpy(img_buffer_host, src, img_size);

    CUDA_CHECK(cudaMemcpyAsync(
        img_buffer_device,
        img_buffer_host,
        img_size,
        cudaMemcpyHostToDevice,
        stream
    ));

    // 3. 计算仿射变换矩阵
    AffineMatrix s2d, d2s; 

    if (mode == MODE_LETTERBOX) {
        // --- YOLO 风格 (Letterbox: 保持比例，居中) ---
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
    } 
    else {
        // --- ResNet 风格 (Stretch: 直接拉伸铺满) ---
        // 忽略原始宽高比
        float scale_x = (float)dst_width / src_width;
        float scale_y = (float)dst_height / src_height;

        s2d.value[0] = scale_x;
        s2d.value[1] = 0;
        s2d.value[2] = 0; // 无偏移
        s2d.value[3] = 0;
        s2d.value[4] = scale_y;
        s2d.value[5] = 0; // 无偏移
    }

    // 4. 计算逆矩阵 (dst -> src)
    // 使用 OpenCV 辅助计算求逆 (保持原有逻辑)
    cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
    cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
    cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);
    memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

    // 5. 启动 Kernel
    int jobs = dst_height * dst_width;
    int threads = 256;
    int blocks = ceil(jobs / (float)threads);

    // 对于 ResNet，填充色可以设为 0，YOLO 设为 114
    // 为了简单，如果不是 YOLO 模式，我们用 0 填充背景（防止有极小黑边）
    uint8_t fill_value = (mode == MODE_LETTERBOX) ? 114 : 0;

    warpaffine_kernel << <blocks, threads, 0, stream >> > (
        img_buffer_device,           
        src_width * 3,               
        src_width,                   
        src_height,                  
        dst,                         
        dst_width,                   
        dst_height,                  
        fill_value,                         
        d2s,                         
        jobs,
        // 传递归一化参数
        use_mean[0], use_mean[1], use_mean[2],
        use_std[0], use_std[1], use_std[2]
    );

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