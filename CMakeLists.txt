cmake_minimum_required(VERSION 3.14.0 FATAL_ERROR)
project(Detections)

# set(ROOT_DIRS "C:/Users/77274/projects")
# set(ROOT_DIRS "D:/projects")
set(Torch_DIR "/usr/local/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS}) # Not needed for CMake >= 2.8.11

message(STATUS "libtorch status:")
message(STATUS "    libraries:${TORCH_LIBRARIES}")
message(STATUS "    include path:${TORCH_INCLUDE_DIRS}")


# set(TorchVision_DIR "${ROOT_DIRS}/3rdpty/torchvision/share/cmake/TorchVision")
# find_package(TorchVision REQUIRED)
# include_directories(${TorchVision_INCLUDE_DIR}) # Not needed for CMake >= 2.8.11
# link_directories("../3rdpty/torchvision/lib")

# message(STATUS "TorchVision status:")
# message(STATUS "    libraries:${TorchVision_LIBRARIES}")
# message(STATUS "    include path:${TorchVision_INCLUDE_DIR}")

# set(OpenCV_DIR "${ROOT_DIRS}/3rdpty/opencv/build/x64/vc15/lib")
# set(OpenCV_DIR "D:/projects/3rdpty/opencv/sources/build/install/x64/vc16/lib")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}) # Not needed for CMake >= 2.8.11


message(STATUS "opencv status:")
message(STATUS "    libraries:${OpenCV_LIBS}")
message(STATUS "    include path:${OpenCV_INCLUDE_DIRS}")

# tensorrt
find_package(TensorRT REQUIRED)
# include_directories("${ROOT_DIRS}/3rdpty/TensorRT-8.5.1.7/include")
# link_directories("${ROOT_DIRS}/3rdpty/TensorRT-8.5.1.7/lib")

# cuda
# include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
# link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib")


# option(WITH_ALGORITHMS "with sample" ON)
# if(WITH_ALGORITHMS)
#    add_subdirectory(algorithms)
# endif()

# option(WITH_SAMPLE "with sample" ON)
# if(WITH_SAMPLE)
#    add_subdirectory(samples)
# endif()


