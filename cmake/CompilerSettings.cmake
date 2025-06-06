# CompilerSettings.cmake - Compiler configuration and flags

# ------------------------------------------------------------------------------
# Basic compiler settings
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Suppress invalid C++ warning from dependencies
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>)
endif()

# ------------------------------------------------------------------------------
# Common compile flags
# ------------------------------------------------------------------------------
set(COMMON_COMPILE_FLAGS
  -Wall
  -Wextra
  -O2
  -fomit-frame-pointer
  -frtti
  -fexceptions
)

# ------------------------------------------------------------------------------
# Debug configuration
# ------------------------------------------------------------------------------
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_definitions(BUILD_DEBUG)
endif()

# ------------------------------------------------------------------------------
# Common definitions
# ------------------------------------------------------------------------------
set(COMMON_DEFINITIONS
  HAVE_STDINT_H
)
