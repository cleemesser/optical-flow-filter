cmake_minimum_required(VERSION 2.8)
project( pVidHistCam )

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


add_executable( pVidHistCam src/pVidHistCam.cpp )
target_link_libraries( pVidHistCam ${OpenCV_LIBS} ${LIBS})
