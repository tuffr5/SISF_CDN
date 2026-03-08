# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Ensure C++17 is enforced

include(FetchContent)
include(CMakePrintHelpers)

include_directories(./ffmpeg_HDF5_filter/src)

# Define the project name
project(nTracer_cdn)

# Architecture-specific optimization flags (GLOBAL - applies to all targets)
message(STATUS "CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    message(STATUS "Using ARM64 flags: -march=armv8-a")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=armv8-a")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    message(STATUS "Using x86-64 flags: -march=x86-64-v3 (AVX2, FMA, F16C)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=x86-64-v3")
else()
    message(WARNING "Unknown processor: ${CMAKE_SYSTEM_PROCESSOR}, using generic -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# Add an executable target
add_executable(nTracer_cdn src/app.cpp)

target_link_libraries(nTracer_cdn pthread)
target_link_libraries(nTracer_cdn sqlite3)

target_link_libraries(nTracer_cdn avcodec)
target_link_libraries(nTracer_cdn avformat)
target_link_libraries(nTracer_cdn avutil)
target_link_libraries(nTracer_cdn swscale)

find_library(ZSTD_LIB zstd PATH ./zstd/lib/)
target_link_libraries(nTracer_cdn "${ZSTD_LIB}")
cmake_print_variables(ZSTD_LIB)

find_library(X264_LIB x264 PATH ./x264/)
target_link_libraries(nTracer_cdn "${X264_LIB}")
cmake_print_variables(X264_LIB)

find_library(CAI_FFMPEG_LIB h5ffmpeg_shared PATH ./ffmpeg_HDF5_filter/)
target_link_libraries(nTracer_cdn "${CAI_FFMPEG_LIB}")
cmake_print_variables(CAI_FFMPEG_LIB)

target_include_directories(nTracer_cdn PRIVATE "/usr/include/hdf5/serial/")

# Linker options
target_link_options(nTracer_cdn PUBLIC -Wl,--no-as-needed -ldl)


FetchContent_Declare(
  tensorstore
  URL "https://github.com/google/tensorstore/archive/refs/tags/v0.1.71.tar.gz"
)

FetchContent_MakeAvailable(tensorstore)


# Define a target that depends on TensorStore...
target_link_libraries(
  nTracer_cdn
#  PRIVATE
    tensorstore::tensorstore tensorstore::all_drivers
)
