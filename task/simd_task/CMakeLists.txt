cmake_minimum_required(VERSION 3.10)
project(simd_task)

set(CMAKE_CXX_STANDARD 11)

# 检测编译器并添加 AVX2 支持
if(MSVC)
    add_compile_options(/arch:AVX2)
else()
    add_compile_options(-mavx2)
endif()

# 定义宏，表示用户已提供实现，不再使用 main.cpp 中的默认空实现
add_definitions(-DUSER_IMPLEMENTATION)

add_executable(simd_task main.cpp solution.cpp)
