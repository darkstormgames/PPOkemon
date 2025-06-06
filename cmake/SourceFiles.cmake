# SourceFiles.cmake - Source file collection

# ------------------------------------------------------------------------------
# Gambatte source files
# ------------------------------------------------------------------------------
file(GLOB_RECURSE GAMBATTE_SOURCES
  ${CMAKE_SOURCE_DIR}/src/libgambatte/*.cpp
  ${CMAKE_SOURCE_DIR}/src/libgambatte/common/*.cpp
  ${CMAKE_SOURCE_DIR}/src/libgambatte/common/resample/src/*.cpp
  ${CMAKE_SOURCE_DIR}/src/libgambatte/common/videolink/*.cpp
  ${CMAKE_SOURCE_DIR}/src/libgambatte/common/videolink/vfilters/*.cpp
  ${CMAKE_SOURCE_DIR}/src/libgambatte/file/*.cpp
  ${CMAKE_SOURCE_DIR}/src/libgambatte/file/unzip/*.cpp
  ${CMAKE_SOURCE_DIR}/src/libgambatte/mem/*.cpp
  ${CMAKE_SOURCE_DIR}/src/libgambatte/mem/mbc/*.cpp
  ${CMAKE_SOURCE_DIR}/src/libgambatte/mem/snes_spc/*.cpp
  ${CMAKE_SOURCE_DIR}/src/libgambatte/sound/*.cpp
  ${CMAKE_SOURCE_DIR}/src/libgambatte/video/*.cpp
)

# ------------------------------------------------------------------------------
# Torch RL source files
# ------------------------------------------------------------------------------
file(GLOB_RECURSE TORCH_RL_SOURCES
  ${CMAKE_SOURCE_DIR}/src/torch/*.cpp
  ${CMAKE_SOURCE_DIR}/src/torch/algorithms/*.cpp
  ${CMAKE_SOURCE_DIR}/src/torch/envs/*.cpp
  ${CMAKE_SOURCE_DIR}/src/torch/networks/*.cpp
  ${CMAKE_SOURCE_DIR}/src/torch/training/*.cpp
  ${CMAKE_SOURCE_DIR}/src/torch/training/callbacks/*.cpp
  ${CMAKE_SOURCE_DIR}/src/torch/training/evaluation/*.cpp
  ${CMAKE_SOURCE_DIR}/src/torch/utils/*.cpp
)

# ------------------------------------------------------------------------------
# Main application source files
# ------------------------------------------------------------------------------
file(READ ${CMAKE_SOURCE_DIR}/src/main.cpp MAIN_CONTENT)
string(STRIP "${MAIN_CONTENT}" MAIN_CONTENT)

if(NOT "${MAIN_CONTENT}" STREQUAL "")
    set(DRL_SOURCES
        ${CMAKE_SOURCE_DIR}/src/main.cpp
        # Add other main application sources here when needed
    )
else()
    set(DRL_SOURCES "")
endif()

# ------------------------------------------------------------------------------
# Test application source files
# ------------------------------------------------------------------------------
set(TEST_SOURCES
    ${CMAKE_SOURCE_DIR}/src/test.cpp
)
