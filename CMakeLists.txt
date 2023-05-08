cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(test LANGUAGES CXX CUDA)

set(CMAKE_GENERATOR "Ninja")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
message("generator is set to ${CMAKE_GENERATOR}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

set(LIB_CUBLAS 
"cudart_static.lib"
"kernel32.lib"
"user32.lib"
"gdi32.lib"
"winspool.lib"
"comdlg32.lib"
"advapi32.lib"
"shell32.lib"
"ole32.lib"
"oleaut32.lib"
"uuid.lib"
"odbc32.lib"
"odbccp32.lib"
"cublas.lib"
)

# 添加头文件路径
include_directories(./include)

# 添加源文件路径
aux_source_directory(./src DIR_SRCS)
add_executable(test ${DIR_SRCS})

target_link_libraries(test ${LIB_CUBLAS} ${OpenMP_CXX_LIBRARIES})

# add_executable(test ./src/test.cu)
target_compile_features(test PUBLIC cxx_std_14)
target_compile_options(test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=/openmp>)

# Required for CUDA builds
set_target_properties(test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)