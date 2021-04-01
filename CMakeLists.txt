cmake_minimum_required(VERSION 3.17)
project(raytracer LANGUAGES CUDA CXX)

# find_package(GTest)

message(STATUS "---")

# ---
# Set paths to sources
# ---

include_directories(src)

set(SRC
    src/main.cu
    src/parse/parser.cu
    src/rendering/engine.cu
    src/scene/camera.cu
    src/scene/sphere.cu
)

set(SRC_TESTS
    # tests/test_example.cc
)

# ---
# Set compilation flags and C++ standard
# ---

set(CXX_OPT_FLAGS "-Ofast -march=native -flto")
# FIXME
set(CUDA_OPT_FLAGS "-O2")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CXX_OPT_FLAGS} -Werror")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3")

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} ${CUDA_OPT_FLAGS}")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -g3 -lineinfo")

# ---
# Detect build type (release or debug)
# ---

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")

    string(REGEX REPLACE ".*/build_" "" BUILD_DIR_NAME ${CMAKE_BINARY_DIR})
    if(${BUILD_DIR_NAME} STREQUAL "debug")
        set(CMAKE_BUILD_TYPE "Debug")
    endif()
endif(NOT CMAKE_BUILD_TYPE)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ---
# Create main executable target
# ---

add_executable(${PROJECT_NAME} ${SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 35 50 72)

message(STATUS "---")
