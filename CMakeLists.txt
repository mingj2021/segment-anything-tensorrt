cmake_minimum_required(VERSION 3.14.0 FATAL_ERROR)
project(Detections)

# add_compile_options(-fno-elide-constructors)
# set(CMAKE_CXX_FLAGS "-fno-elide-constructors ${CMAKE_CXX_FLAGS}")
# set(ROOT_DIRS "C:/Users/77274/projects")
# set(ROOT_DIRS "D:/projects")
# set( CMAKE_CXX_COMPILER /usr/bin/g++ )
# set(CMAKE_CXX_FLAGS "-fno-elide-constructors ${CMAKE_CXX_FLAGS}")
set(Torch_DIR "/usr/local/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
# include_directories(${TORCH_INCLUDE_DIRS}) # Not needed for CMake >= 2.8.11


# set(TorchVision_DIR "${ROOT_DIRS}/3rdpty/torchvision/share/cmake/TorchVision")
# find_package(TorchVision REQUIRED)
# include_directories(${TorchVision_INCLUDE_DIR}) # Not needed for CMake >= 2.8.11
# link_directories("../3rdpty/torchvision/lib")

find_package(OpenCV REQUIRED)

set(SAMPLE_SOURCES main.cpp)
set(TARGET_NAME sampleSAM)

set(SAMPLE_DEP_LIBS
    nvinfer
    nvonnxparser
)

# commons
include_directories("./commons")

add_executable(${TARGET_NAME} ${SAMPLE_SOURCES})
target_link_libraries(${TARGET_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS})
# target_link_libraries(${TARGET_NAME} TorchVision::TorchVision)
target_link_libraries(${TARGET_NAME} ${SAMPLE_DEP_LIBS})


set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)

