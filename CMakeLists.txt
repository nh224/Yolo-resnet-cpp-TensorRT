cmake_minimum_required(VERSION 3.18)

# 项目声明
project(YOLOv11TRT LANGUAGES CXX CUDA)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 允许覆盖 TensorRT 根目录
if(NOT DEFINED TENSORRT_ROOT)
    set(TENSORRT_ROOT "/usr/src/tensorrt" CACHE PATH "TensorRT root directory")
endif()

# =============================================================================
# 1. 查找依赖 (针对 Jetson aarch64 优化)
# =============================================================================

# 查找 TensorRT 头文件
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS
        ${TENSORRT_ROOT}/include
        /usr/include/aarch64-linux-gnu  # Jetson 标准路径
        /usr/include
)

# 查找 NvInfer 库
find_library(NVINFER_LIBRARY NAMES nvinfer libnvinfer
    HINTS
        ${TENSORRT_ROOT}/lib
        /usr/lib/aarch64-linux-gnu      # Jetson 标准路径
        /usr/lib
)

# 查找 NvOnnxParser 库 (Convert 模式必需)
find_library(NVONNXPARSER_LIBRARY NAMES nvonnxparser libnvonnxparser
    HINTS
        ${TENSORRT_ROOT}/lib
        /usr/lib/aarch64-linux-gnu
        /usr/lib
)

# 检查是否找到
if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIBRARY)
    message(FATAL_ERROR "TensorRT not found. Check paths.")
endif()

message(STATUS "Found TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "Found TensorRT libs: ${NVINFER_LIBRARY} ${NVONNXPARSER_LIBRARY}")

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 查找 CUDA
find_package(CUDAToolkit REQUIRED)

# =============================================================================
# 2. 源文件定义 (关键部分)
# =============================================================================

# 包含目录
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# 指定源文件列表
# 注意：根据您的日志，main.cpp 在根目录，wrapper 在 src 目录
set(SOURCES
    main.cpp              
    src/wrapper.cpp       
    src/preprocess.cu
)

# =============================================================================
# 3. 目标构建配置
# =============================================================================

# 创建可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

# 链接库
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    CUDA::cudart
    ${NVINFER_LIBRARY}
    ${NVONNXPARSER_LIBRARY} # 链接 ONNX 解析器
)

# CUDA 编译选项
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 设置 CUDA 架构 (Orin = 87, Xavier = 72, RTX30xx = 86)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87 CACHE STRING "CUDA architectures to build for")
endif()

# 输出目录设为 build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

message(STATUS "Build setup complete for Jetson Orin (Arch 87)")