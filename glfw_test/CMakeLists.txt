cmake_minimum_required(VERSION 3.3)
project(glfw_test)

include_directories(${glfw_INCLUDE_DIRS} "${GLFW_SOURCE_DIR}/deps")

link_libraries("${MATH_LIBRARY}")

find_package( OpenGL REQUIRED )
include_directories(${OPENGL_INCLUDE_DIR})

LIST(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})
find_package(GLFW REQUIRED)
include_directories(${GLFW_INCLUDE_DIRS})

find_package( GLEW REQUIRED )
include_directories(${GLEW_INCLUDE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(glfw_test ../glfw_test.cpp)
target_link_libraries(glfw_test ${GLFW_LIBRARY}
							${OPENGL_LIBRARIES}
							${GLEW_LIBRARIES})