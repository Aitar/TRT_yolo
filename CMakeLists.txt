cmake_minimum_required(VERSION 3.22)
project(yolo CUDA)

set(CMAKE_CUDA_STANDARD 14)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

set(CUDNN_INCLUDE_DIR )
set(CUDNN_LIBRARY )

set(CUDA_INCLUDE /usr/local/cuda-11.3/include)
set(CUDA_LIB /usr/local/cuda-11.3/lib64)

set(CV_DIR /opt/miniconda3/envs/clibs/include/opencv4)
set(CV_LIB /opt/miniconda3/envs/clibs/lib)

set(TRT_INCLUDE )
set(TRT_LIB )

include_directories(${CV_DIR})
link_directories(${CV_LIB})

include_directories(${TRT_INCLUDE})
link_directories(${TRT_LIB})

include_directories(${CUDNN_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE})

link_directories(${CUDNN_LIBRARY})
link_directories(${CUDA_LIB})

add_executable(yolo main.cu tensor.h utils.h context.h yolo.h kernel.cuh)

target_link_libraries(yolo cudart nvonnxparser nvinfer opencv_core opencv_video opencv_videoio opencv_imgproc cudnn cublas cublasLt)

set_target_properties(yolo PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
