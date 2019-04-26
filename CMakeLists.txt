cmake_minimum_required(VERSION 3.5)
project(weighted_multi_projection)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "-O0 -std=gnu++11 -fopenmp -fexceptions")

include_directories(
  include
)

add_executable(weighted_multi_projection 
                src/main.cpp
                src/ParameterManager.cpp
              )
              
find_package(PkgConfig REQUIRED)
find_package(PCL REQUIRED)
find_package(VTK REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
target_link_libraries(weighted_multi_projection ${PCL_LIBRARIES})
include_directories(${VTK_INCLUDE_DIRS})
target_link_libraries(weighted_multi_projection ${VTK_LIBRARIES})

