cmake_minimum_required(VERSION 3.3)

project(opencv_example_project)

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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
        ../aruco_ar.cpp
        ../shader.cpp
        ../controls.cpp
        ../texture.cpp
        ../objloader.cpp)
add_executable(AR ${SOURCE_FILES})
target_link_libraries(AR
        ${OPENGL_LIBRARIES}
        ${OpenCV_LIBS}
        ${GLEW_LIBRARIES}
        ${GLFW_LIBRARY}
        ${PNG_LIBRARY})

#add_executable(detect_marker ../detect_marker.cpp)
#target_link_libraries(detect_marker ${OpenCV_LIBS})

#add_executable(create_board ../create_board.cpp)
#target_link_libraries(create_board ${OpenCV_LIBS})

#add_executable(calibrate_camera ../calibrate_camera.cpp)
#target_link_libraries(calibrate_camera ${OpenCV_LIBS})