cmake_minimum_required(VERSION 3.3)

project(ORB_SLAM2_AR)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(OpenCV REQUIRED)
find_package( OpenGL REQUIRED )
include_directories(${OPENGL_INCLUDE_DIR})

find_package( GLM REQUIRED )
include_directories(${GLM_INCLUDE_DIR})

find_package( GLEW REQUIRED )
include_directories(${GLEW_INCLUDE_DIR})
#include(${CMAKE_BINARY_DIR}/CMakeTools/CMakeToolsHelpers.cmake)

find_package( PNG REQUIRED )
include_directories(${PNG_INCLUDE_DIR})

LIST(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})
find_package(GLFW REQUIRED)
include_directories(${GLFW_INCLUDE_DIRS})

if(CMAKE_VERSION VERSION_LESS "2.8.11")
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)#baitao
include_directories(${Eigen_INCLUDE_DIRS})

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
        ../src/ar_gl.cpp
        ../src/shader.cpp
        ../src/texture.cpp
        ../src/objloader.cpp
        ../src/plane_detect.cpp)
add_executable(ORB_AR ${SOURCE_FILES})
target_link_libraries(ORB_AR
        ${OPENGL_LIBRARIES}
        ${OpenCV_LIBS}
        ${GLEW_LIBRARIES}
        ${GLFW_LIBRARY}
        ${PNG_LIBRARY}
        ${EIGEN3_LIBS}
        ${Pangolin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.dylib
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.dylib
        ${PROJECT_SOURCE_DIR}/lib/libORB_SLAM2.dylib)
