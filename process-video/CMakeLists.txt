cmake_minimum_required(VERSION 2.8)
project( processVideo )

find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED)

# Required libraries
# It assumes flowfilter_gpu is installed at /usr/local/lib
set( LIBS flowfilter_gpu)

include_directories(${CUDA_INCLUDE_DIRS})

#################################################
# COMPILER SETTINGS
#################################################
set (CMAKE_CXX_COMPILER         "g++")
set (CMAKE_CXX_FLAGS            "-std=c++11 -flto -O2 -Wall")


add_executable( processVideo src/processVideo.cpp )
target_link_libraries( processVideo ${OpenCV_LIBS} ${LIBS})

add_executable( processVideo src/pVidHist.cpp )
target_link_libraries( pVidHist ${OpenCV_LIBS} ${LIBS})
