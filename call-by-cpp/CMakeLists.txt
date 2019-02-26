
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(call-by-cpp)

set(OpenCV_DIR "/data/qiuyq/soft/opencv3.4/opencv-master/build")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(Torch REQUIRED)

add_executable(call call.cpp)
target_link_libraries(call "${TORCH_LIBRARIES}" ${OpenCV_LIBS})
set_property(TARGET call PROPERTY CXX_STANDARD 11)

