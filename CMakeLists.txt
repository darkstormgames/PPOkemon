cmake_minimum_required(VERSION 3.18)

project(DRLGambatteTest)

# Suppress invalid C++ warning from dependencies
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>)
endif()

set(CMAKE_CXX_STANDARD 17)
option(USE_WINDOWS "Use Windows as OS" OFF)
option(USE_CUDA "Use CUDA for GPU acceleration" ON)
option(USE_ROCM "Use ROCm for GPU acceleration" OFF)
option(USE_SDL2 "Use SDL2 for renderer" OFF)
option(STATIC_LINK_LIBS "Link SDL2, zlib, and minizip statically (auto-download/build)" ON)

include(FetchContent)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(USE_CUDA)
  set(CMAKE_CUDA_ARCHITECTURES "75") # RTX 20-Series
  set(PYTORCH_VERSION "2.7")
  set(PYTORCH_CUDA_VERSION "12.9")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  # assumes find_package(CUDAToolkit) was already done
  if(
    PYTORCH_VERSION VERSION_GREATER_EQUAL 2.5.0 AND
    PYTORCH_CUDA_VERSION VERSION_GREATER_EQUAL 12
  )
    message(STATUS "PyTorch NVTX headers workaround: Yes")
    if(NOT TARGET CUDA::nvToolsExt AND TARGET CUDA::nvtx3)
      add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
      target_compile_definitions(
        CUDA::nvToolsExt INTERFACE
        TORCH_CUDA_USE_NVTX3
      )
      target_link_libraries(CUDA::nvToolsExt INTERFACE CUDA::nvtx3)
    endif()
  else()
    message(STATUS "PyTorch NVTX headers workaround: No")
  endif()
endif()

# ------------------------------------------------------------------------------
# SDL2
# ------------------------------------------------------------------------------
if(USE_SDL2)
  if(STATIC_LINK_LIBS)
    set(SDL2_STATIC ON)
    FetchContent_Declare(
      SDL2
      GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
      GIT_TAG release-2.28.5 # or latest stable tag
    )
    FetchContent_MakeAvailable(SDL2)
  else()
    find_package(SDL2 REQUIRED)
  endif()
  add_definitions(-DUSE_SDL2)
endif()

# ------------------------------------------------------------------------------
# zlib
# ------------------------------------------------------------------------------
if(STATIC_LINK_LIBS)
  FetchContent_Declare(
    zlib
    URL https://zlib.net/zlib-1.3.1.tar.gz
  )
  FetchContent_MakeAvailable(zlib)
endif()

# ------------------------------------------------------------------------------
# minizip
# ------------------------------------------------------------------------------
if(STATIC_LINK_LIBS)
  FetchContent_Declare(
    minizip
    GIT_REPOSITORY https://github.com/zlib-ng/minizip-ng.git
    GIT_TAG 4.0.6 # or latest stable
  )
  set(MZ_FETCH_LIBZ ON CACHE BOOL "" FORCE)
  set(MZ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(minizip)
endif()

# ------------------------------------------------------------------------------
# Include paths (TODO: Check, which of these are actually needed)
# ------------------------------------------------------------------------------
include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include/config
  ${CMAKE_SOURCE_DIR}/include/libgambatte
  ${CMAKE_SOURCE_DIR}/include/libgambatte/common
  ${CMAKE_SOURCE_DIR}/include/libgambatte/common/resample
  ${CMAKE_SOURCE_DIR}/include/libgambatte/common/resample/src
  ${CMAKE_SOURCE_DIR}/include/libgambatte/common/videolink
  ${CMAKE_SOURCE_DIR}/include/libgambatte/common/videolink/vfilters
  ${CMAKE_SOURCE_DIR}/include/libgambatte/file
  ${CMAKE_SOURCE_DIR}/include/libgambatte/file/unzip
  ${CMAKE_SOURCE_DIR}/include/libgambatte/mem
  ${CMAKE_SOURCE_DIR}/include/libgambatte/mem/mbc
  ${CMAKE_SOURCE_DIR}/include/libgambatte/mem/snes_spc
  ${CMAKE_SOURCE_DIR}/include/libgambatte/sound
  ${CMAKE_SOURCE_DIR}/include/libgambatte/video
  ${CMAKE_SOURCE_DIR}/src/common
  ${CMAKE_SOURCE_DIR}/src/common/resample
  ${CMAKE_SOURCE_DIR}/src/common/resample/src
  ${CMAKE_SOURCE_DIR}/src/common/videolink
  ${CMAKE_SOURCE_DIR}/src/common/videolink/vfilters
  ${CMAKE_SOURCE_DIR}/src/emulation
  ${CMAKE_SOURCE_DIR}/src/environments
  ${CMAKE_SOURCE_DIR}/src/libgambatte
  # ${CMAKE_SOURCE_DIR}/src/libgambatte/file
  # ${CMAKE_SOURCE_DIR}/src/libgambatte/file/unzip
  # ${CMAKE_SOURCE_DIR}/src/libgambatte/mem
  # ${CMAKE_SOURCE_DIR}/src/libgambatte/mem/mbc
  # ${CMAKE_SOURCE_DIR}/src/libgambatte/mem/snes_spc
  # ${CMAKE_SOURCE_DIR}/src/libgambatte/sound
  # ${CMAKE_SOURCE_DIR}/src/libgambatte/video
  ${CMAKE_SOURCE_DIR}/src/torch_rl
  ${CMAKE_SOURCE_DIR}/src/torch_rl/envs
  ${CMAKE_SOURCE_DIR}/src/torch_rl/rl
  ${CMAKE_SOURCE_DIR}/src/torch_rl/utils
)

# ------------------------------------------------------------------------------
# Source files
# ------------------------------------------------------------------------------
file(GLOB_RECURSE GAMBATTE_SOURCES
  ${CMAKE_SOURCE_DIR}/src/libgambatte/common/*.cpp
  ${CMAKE_SOURCE_DIR}/src/libgambatte/common/resample/src/*.cpp
  ${CMAKE_SOURCE_DIR}/src/libgambatte/common/videolink/*.cpp
  ${CMAKE_SOURCE_DIR}/src/libgambatte/common/videolink/vfilters/*.cpp
  ${CMAKE_SOURCE_DIR}/src/libgambatte/*.cpp
  ${CMAKE_SOURCE_DIR}/src/libgambatte/file/*.cpp
  ${CMAKE_SOURCE_DIR}/src/libgambatte/file/unzip/*.cpp
  ${CMAKE_SOURCE_DIR}/src/libgambatte/mem/*.cpp
  ${CMAKE_SOURCE_DIR}/src/libgambatte/mem/mbc/*.cpp
  ${CMAKE_SOURCE_DIR}/src/libgambatte/mem/snes_spc/*.cpp
  ${CMAKE_SOURCE_DIR}/src/libgambatte/sound/*.cpp
  ${CMAKE_SOURCE_DIR}/src/libgambatte/video/*.cpp
)

file(GLOB_RECURSE TORCH_RL_SOURCES
  ${CMAKE_SOURCE_DIR}/src/torch_rl/*.cpp
  ${CMAKE_SOURCE_DIR}/src/torch_rl/envs/*.cpp
  ${CMAKE_SOURCE_DIR}/src/torch_rl/rl/*.cpp
  ${CMAKE_SOURCE_DIR}/src/torch_rl/utils/*.cpp
)

set(DRL_SOURCES
  ${CMAKE_SOURCE_DIR}/src/main.cpp
  ${CMAKE_SOURCE_DIR}/src/emulation/emu_gbc.cpp
  ${CMAKE_SOURCE_DIR}/src/emulation/renderer.cpp
  ${CMAKE_SOURCE_DIR}/src/emulation/inputhandler.cpp
  ${CMAKE_SOURCE_DIR}/src/environments/pokered_v1.cpp
)

find_package(Torch QUIET
  HINTS
  ${CMAKE_SOURCE_DIR}/../libtorch
  ${CMAKE_SOURCE_DIR}/libtorch
  ${CMAKE_SOURCE_DIR}/../libtorch/lib
  ${CMAKE_SOURCE_DIR}/../libtorch/include)

if(NOT Torch_FOUND)
  message(STATUS "Torch not found. Downloading libtorch...")

  # Determine the download URL based on platform and options
  set(LIBTORCH_URL "")
  set(LIBTORCH_DOWNLOAD_DIR "${CMAKE_SOURCE_DIR}/libtorch_download")
  set(LIBTORCH_EXTRACT_DIR "${CMAKE_SOURCE_DIR}")

  if(USE_WINDOWS)
    if(USE_CUDA)
      set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-2.7.0%2Bcu128.zip")
    else()
      set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.7.0%2Bcpu.zip")
    endif()
    if(USE_ROCM)
      message(FATAL_ERROR "ROCm is not supported on Windows")
    endif()
  else()
    # Linux
    if(USE_CUDA)
      set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu128.zip")
    elseif(USE_ROCM)
      set(LIBTORCH_URL "https://download.pytorch.org/libtorch/rocm6.3/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Brocm6.3.zip")
    else()
      set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcpu.zip")
    endif()
  endif()

  # Download libtorch
  file(MAKE_DIRECTORY ${LIBTORCH_DOWNLOAD_DIR})
  set(LIBTORCH_ZIP "${LIBTORCH_DOWNLOAD_DIR}/libtorch.zip")

  if(NOT EXISTS ${LIBTORCH_ZIP})
    message(STATUS "Downloading libtorch from ${LIBTORCH_URL}")
    file(DOWNLOAD
      ${LIBTORCH_URL}
      ${LIBTORCH_ZIP}
      SHOW_PROGRESS
      STATUS DOWNLOAD_STATUS
    )

    list(GET DOWNLOAD_STATUS 0 DOWNLOAD_RESULT)
    if(NOT DOWNLOAD_RESULT EQUAL 0)
      file(REMOVE ${LIBTORCH_ZIP})
      message(FATAL_ERROR "Failed to download libtorch. Error: ${DOWNLOAD_STATUS}")
    endif()
  endif()

  # Extract libtorch
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/libtorch")
    message(STATUS "Extracting libtorch...")
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xzf ${LIBTORCH_ZIP}
      WORKING_DIRECTORY ${LIBTORCH_EXTRACT_DIR}
      RESULT_VARIABLE EXTRACT_RESULT
    )

    if(NOT EXTRACT_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to extract libtorch")
    endif()
  endif()

  # Try to find Torch again
  find_package(Torch REQUIRED
    HINTS
    ${CMAKE_SOURCE_DIR}/libtorch
    ${CMAKE_SOURCE_DIR}/libtorch/lib
    ${CMAKE_SOURCE_DIR}/libtorch/include)
endif()

if(Torch_FOUND)
  include_directories(${TORCH_INCLUDE_DIRS})
  link_directories(${TORCH_LIBRARY_DIRS})
  add_definitions(-DUSE_TORCH)
  message(STATUS "Found Torch: ${TORCH_LIBRARIES}")
else()
  message(FATAL_ERROR "Torch not found even after download attempt. Please check your configuration.")
endif()

# Function to recursively clean compile options from a target
function(clean_target_options target)
  if(TARGET ${target})
    get_target_property(interface_options ${target} INTERFACE_COMPILE_OPTIONS)
    if(interface_options)
      list(REMOVE_ITEM interface_options "-Wno-duplicate-decl-specifier")
      set_target_properties(${target} PROPERTIES INTERFACE_COMPILE_OPTIONS "${interface_options}")
    endif()

    get_target_property(interface_link_libs ${target} INTERFACE_LINK_LIBRARIES)
    if(interface_link_libs)
      foreach(lib ${interface_link_libs})
        if(TARGET ${lib})
          clean_target_options(${lib})
        endif()
      endforeach()
    endif()
  endif()
endfunction()

# Clean all torch-related targets
foreach(torch_target torch torch_cpu torch_cuda c10 c10_cuda caffe2_nvrtc torch_hip c10_hip)
  clean_target_options(${torch_target})
endforeach()

# Remove problematic flags from CMAKE_CXX_FLAGS
string(REPLACE "-Wno-duplicate-decl-specifier" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "-Wno-duplicate-decl-specifier" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

# Also check and clean flags that might come from dependencies
if(TORCH_CXX_FLAGS)
  string(REPLACE "-Wno-duplicate-decl-specifier" "" TORCH_CXX_FLAGS "${TORCH_CXX_FLAGS}")
endif()

# ------------------------------------------------------------------------------
# Executable
# ------------------------------------------------------------------------------
add_executable(drl_test ${DRL_SOURCES} ${GAMBATTE_SOURCES} ${TORCH_RL_SOURCES})

# Clean compile options from the target after it's created
get_target_property(DRL_TEST_COMPILE_OPTIONS drl_test COMPILE_OPTIONS)
if(DRL_TEST_COMPILE_OPTIONS)
  list(REMOVE_ITEM DRL_TEST_COMPILE_OPTIONS "-Wno-duplicate-decl-specifier")
  set_property(TARGET drl_test PROPERTY COMPILE_OPTIONS ${DRL_TEST_COMPILE_OPTIONS})
endif()

# Also clean inherited compile options
get_target_property(DRL_TEST_INTERFACE_COMPILE_OPTIONS drl_test INTERFACE_COMPILE_OPTIONS)
if(DRL_TEST_INTERFACE_COMPILE_OPTIONS)
  list(REMOVE_ITEM DRL_TEST_INTERFACE_COMPILE_OPTIONS "-Wno-duplicate-decl-specifier")
  set_property(TARGET drl_test PROPERTY INTERFACE_COMPILE_OPTIONS ${DRL_TEST_INTERFACE_COMPILE_OPTIONS})
endif()

# Apply -frtti and -fexceptions only to files in src/
foreach(source_file IN LISTS DRL_SOURCES)
  get_filename_component(src_filename ${source_file} NAME)
  set_source_files_properties(${source_file} PROPERTIES
    COMPILE_FLAGS "-frtti -fexceptions"
  )
endforeach()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_definitions(BUILD_DEBUG)
endif()

target_compile_definitions(drl_test PRIVATE HAVE_STDINT_H)
target_compile_options(drl_test PRIVATE -Wall -Wextra -O2 -fomit-frame-pointer)

# Suppress specific compiler warnings
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # For GCC, we need to use -Wno-error to suppress specific warnings
  target_compile_options(drl_test PRIVATE -Wno-error=unknown-pragmas)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # For Clang, use -Wno-unknown-warning-option
  target_compile_options(drl_test PRIVATE -Wno-unknown-warning-option)
endif()

# ------------------------------------------------------------------------------
# Link libraries
# ------------------------------------------------------------------------------
if(USE_SDL2)
  if(STATIC_LINK_LIBS)
    target_link_libraries(drl_test PRIVATE SDL2-static)
  else()
    target_link_libraries(drl_test PRIVATE SDL2::SDL2)
  endif()
endif()

if(STATIC_LINK_LIBS)
  target_link_libraries(drl_test PRIVATE zlibstatic minizip)
else()
  find_package(ZLIB REQUIRED)
  find_library(MINIZIP_LIB minizip REQUIRED)
  target_link_libraries(drl_test PRIVATE ZLIB::ZLIB ${MINIZIP_LIB})
endif()

# Create a custom interface library to wrap torch with cleaned flags
add_library(torch_clean INTERFACE)
target_link_libraries(torch_clean INTERFACE ${TORCH_LIBRARIES})
get_target_property(TORCH_CLEAN_OPTIONS torch_clean INTERFACE_COMPILE_OPTIONS)
if(TORCH_CLEAN_OPTIONS)
  list(REMOVE_ITEM TORCH_CLEAN_OPTIONS "-Wno-duplicate-decl-specifier")
  set_property(TARGET torch_clean PROPERTY INTERFACE_COMPILE_OPTIONS ${TORCH_CLEAN_OPTIONS})
endif()

target_link_libraries(drl_test PRIVATE torch_clean)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(drl_test PUBLIC OpenMP::OpenMP_CXX)
endif()

# ------------------------------------------------------------------------------
# RPATH (for dynamic builds)
# ------------------------------------------------------------------------------
set_target_properties(drl_test PROPERTIES
  BUILD_RPATH ${CMAKE_BINARY_DIR}
  INSTALL_RPATH ${CMAKE_BINARY_DIR}
)
