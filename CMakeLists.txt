cmake_minimum_required(VERSION 3.17)
project(raytracer LANGUAGES CUDA CXX)

find_package(GTest)

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
    src/scene/plan.cu
    src/scene/raybox.cu
    src/scene/sphere.cu
    src/scene/triangle.cu
)

set(SRC_TESTS
    tests/test_image.cu
)

# ---
# Set compilation flags and C++ standard
# ---

set(CXX_OPT_FLAGS "-Ofast -march=native -flto")
# FIXME
set(CUDA_OPT_FLAGS "")

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

# ---
# Create targets for unit tests
# ---

if(${GTEST_FOUND})
    message(STATUS "GTest enabled: unit tests will be generated")
    enable_testing()

    # For each test file create an executable of test. Launched using ctest
    foreach(f IN LISTS SRC_TESTS)
        # Find the test_name by deleting the extension and the parent_dir
        string(REGEX REPLACE "\\.[^.]*$" "" F_WITHOUT_EXT ${f})
        string(REGEX REPLACE "[^/]*/" "" TEST_NAME ${F_WITHOUT_EXT})

        add_executable(${TEST_NAME} ${f})

        set_target_properties(${TEST_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET ${TEST_NAME} PROPERTY CUDA_ARCHITECTURES 35 50 72)
        target_link_libraries(${TEST_NAME}
            PRIVATE
            ${GTEST_BOTH_LIBRARIES}
            ${CMAKE_THREAD_LIBS_INIT})

        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
        set_tests_properties(${TEST_NAME} PROPERTIES TIMEOUT 2)
    endforeach()
else()
    message(STATUS "GTest disabled: unit tests will not be generated")
endif()

message(STATUS "---")
