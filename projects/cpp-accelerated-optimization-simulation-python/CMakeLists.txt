cmake_minimum_required(VERSION 3.15...4.4)
project(cpp_accelerated_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_core src/cpp_accelerated_sim/_core.cpp)
target_include_directories(_core PRIVATE src/cpp_accelerated_sim)

if(MSVC)
  target_compile_options(_core PRIVATE /O2)
else()
  target_compile_options(_core PRIVATE -O3)
endif()

install(TARGETS _core LIBRARY DESTINATION cpp_accelerated_sim)
