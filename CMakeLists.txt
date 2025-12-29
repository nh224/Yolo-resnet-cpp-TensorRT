cmake_minimum_required(VERSION 3.18)

# Project declaration with C++ and CUDA support
project(YOLOv11TRT LANGUAGES CXX CUDA)

# Set C++ standard to C++17（与VS Code配置一致）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ========== 手动指定TensorRT路径（apt安装的标准路径） ==========
# TensorRT头文件路径（apt安装固定路径）
set(TENSORRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu")
# TensorRT库文件路径（apt安装固定路径）
set(TENSORRT_LIB_DIR "/usr/lib/x86_64-linux-gnu")

# 查找OpenCV（对应VS Code的/usr/include/opencv4）
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found! 执行：sudo apt-get install libopencv-dev")
endif()

# 查找CUDA（现代CMake方式，对应VS Code的/usr/local/cuda/include）
find_package(CUDAToolkit REQUIRED)
if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA not found! 执行：sudo apt-get install nvidia-cuda-toolkit")
endif()

# 验证TensorRT头文件是否存在
if(NOT EXISTS "${TENSORRT_INCLUDE_DIR}/NvInfer.h")
    message(FATAL_ERROR "TensorRT头文件未找到！执行：sudo apt-get install libnvinfer-dev libnvonnxparser-dev")
endif()

# 验证TensorRT库文件是否存在
if(NOT EXISTS "${TENSORRT_LIB_DIR}/libnvinfer.so")
    message(FATAL_ERROR "TensorRT库文件未找到！执行：sudo apt-get install libnvinfer8 libnvonnxparser8")
endif()

# 包含目录（全部使用系统标准路径）
include_directories(
    ${CMAKE_SOURCE_DIR}/include  # 项目自身头文件
    ${CMAKE_SOURCE_DIR}/src      # 源码目录
    ${TENSORRT_INCLUDE_DIR}      # TensorRT头文件（apt安装路径）
    ${CUDAToolkit_INCLUDE_DIRS}  # CUDA头文件
    ${OpenCV_INCLUDE_DIRS}       # OpenCV头文件
    /usr/include/x86_64-linux-gnu  # 系统架构头文件
)

# 定义源文件
set(SOURCES
    main.cpp
    src/yolov11.cpp
    src/preprocess.cu
)

# 创建可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

# 定义API_EXPORTS宏
target_compile_definitions(${PROJECT_NAME} PRIVATE API_EXPORTS)

# 目标包含目录（补充）
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# ========== 手动链接TensorRT库 ==========
set(TENSORRT_LIBS
    ${TENSORRT_LIB_DIR}/libnvinfer.so
    ${TENSORRT_LIB_DIR}/libnvonnxparser.so
    #${TENSORRT_LIB_DIR}/libnvparsers.so
    ${TENSORRT_LIB_DIR}/libnvinfer_plugin.so
)

# 链接所有依赖库
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}                # OpenCV库
    CUDA::cudart                  # CUDA库（现代方式）
    ${TENSORRT_LIBS}              # TensorRT库（手动指定）
)

# CUDA相关配置
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON  # 启用CUDA分离编译
)
# 指定CUDA架构（根据你的GPU调整，例如RTX30/40系列填86，GTX10系列填61）
set(CMAKE_CUDA_ARCHITECTURES 86)

# 可执行文件输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)