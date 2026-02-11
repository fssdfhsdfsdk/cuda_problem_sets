cmake_minimum_required(VERSION 3.18)
project(simt_task LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_executable(simt_task 
    main.cpp
    framework.cu
    kernel.cu  # 用户必须修改此文件
)

# 强制要求用户修改kernel.cu（初始含缺陷实现）
add_custom_target(check_implementation
    COMMAND ${CMAKE_COMMAND} -E echo "⚠️  检测到未修改的kernel.cu!"
    COMMAND grep -q "FIXME: 此实现" kernel.cu && exit 1 || exit 0
    VERBATIM
)
add_dependencies(simt_task check_implementation)