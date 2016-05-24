cmake_minimum_required(VERSION 2.8)
#project(project)

# DART installation
include_directories(/usr/local/include/dart/)

# LCM messages
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/bot_core)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(lcm_depth_provider STATIC lcm_depth_provider.cpp)
