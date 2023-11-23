cmake_minimum_required(VERSION 3.22.1)
project(yolo CUDA)

set(CMAKE_CUDA_STANDARD 14)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

set(CUDNN_INCLUDE_DIR /home/special/miniconda3/envs/huangdingli_cuda/cudnn-8.9.4.25/include)
set(CUDNN_LIBRARY /home/special/miniconda3/envs/huangdingli_cuda/cudnn-8.9.4.25/lib)

set(CV_DIR /home/special/miniconda3/envs/huangdingli_cuda/include/opencv4)
set(CV_LIB /home/special/miniconda3/envs/huangdingli_cuda/lib)

set(TRT_INCLUDE /home/special/miniconda3/envs/huangdingli_cuda/TensorRT-8.6.1.6/include)
set(TRT_LIB /home/special/miniconda3/envs/huangdingli_cuda/TensorRT-8.6.1.6/lib)

include_directories(${CV_DIR})
link_directories(${CV_LIB})

include_directories(${TRT_INCLUDE})
link_directories(${TRT_LIB})

include_directories(${CUDNN_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE})

link_directories(${CUDNN_LIBRARY})
link_directories(${CUDA_LIB})

add_executable(yolo main.cu tensor.h utils.h context.h yolo.h kernel.cuh)

target_link_libraries(yolo cudart nvonnxparser nvinfer_plugin nvinfer opencv_core opencv_video opencv_videoio opencv_imgproc cudnn cublas cublasLt)

set_target_properties(yolo PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
