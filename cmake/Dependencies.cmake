# Dependencies.cmake - External dependency management

# ------------------------------------------------------------------------------
# CUDA Configuration
# ------------------------------------------------------------------------------
if(USE_CUDA)
  set(CMAKE_CUDA_ARCHITECTURES "75") # RTX 20-Series
  set(PYTORCH_VERSION "2.7")
  set(PYTORCH_CUDA_VERSION "12.9")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  # PyTorch NVTX headers workaround
  if(PYTORCH_VERSION VERSION_GREATER_EQUAL 2.5.0 AND PYTORCH_CUDA_VERSION VERSION_GREATER_EQUAL 12)
    message(STATUS "PyTorch NVTX headers workaround: Yes")
    if(NOT TARGET CUDA::nvToolsExt AND TARGET CUDA::nvtx3)
      add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
      target_compile_definitions(CUDA::nvToolsExt INTERFACE TORCH_CUDA_USE_NVTX3)
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
      GIT_TAG release-2.28.5
    )
    FetchContent_MakeAvailable(SDL2)
  else()
    find_package(SDL2 REQUIRED)
  endif()
  add_definitions(-DUSE_SDL2)
endif()

# ------------------------------------------------------------------------------
# zlib and minizip
# ------------------------------------------------------------------------------
if(STATIC_LINK_LIBS)
  FetchContent_Declare(
    zlib
    URL https://zlib.net/zlib-1.3.1.tar.gz
  )
  FetchContent_MakeAvailable(zlib)

  FetchContent_Declare(
    minizip
    GIT_REPOSITORY https://github.com/zlib-ng/minizip-ng.git
    GIT_TAG 4.0.6
  )
  set(MZ_FETCH_LIBZ ON CACHE BOOL "" FORCE)
  set(MZ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(minizip)
else()
  find_package(ZLIB REQUIRED)
  find_library(MINIZIP_LIB minizip REQUIRED)
endif()

# ------------------------------------------------------------------------------
# PyTorch/LibTorch
# ------------------------------------------------------------------------------
include(${CMAKE_CURRENT_LIST_DIR}/LibTorch.cmake)

# ------------------------------------------------------------------------------
# OpenMP
# ------------------------------------------------------------------------------
find_package(OpenMP)
