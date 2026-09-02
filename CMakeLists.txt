cmake_minimum_required(VERSION 3.20)

project(CudaLearning LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(
    CudaLearning
    main.cpp
    timer.cpp
    chapter01/hellocuda.cu
    chapter02/SumArraysOnGPU.cu
    chapter02/SumMatrixOnGPU.cu
    chapter02/ParallelReductionWrapDivergence.cu
)

target_include_directories(CudaLearning PRIVATE ${CMAKE_SOURCE_DIR})