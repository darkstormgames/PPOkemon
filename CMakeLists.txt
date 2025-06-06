cmake_minimum_required(VERSION 3.18)

project(DRLGambatteTest)

# ------------------------------------------------------------------------------
# Build Options
# ------------------------------------------------------------------------------
option(USE_WINDOWS "Use Windows as OS" OFF) # not supported yet
option(USE_CUDA "Use CUDA for GPU acceleration" ON)
option(USE_ROCM "Use ROCm for GPU acceleration" OFF)
option(USE_SDL2 "Use SDL2 for renderer" OFF)
option(STATIC_LINK_LIBS "Link SDL2, zlib, and minizip statically (auto-download/build)" OFF)
option(BUILD_MAIN_APP "Build main PPOkemon application" OFF)
option(BUILD_TEST_APP "Build basic test executable" ON)
option(BUILD_POKERED_ROM "Automatically build Pokemon Red ROM for testing" ON)

# ------------------------------------------------------------------------------
# Include CMake modules
# ------------------------------------------------------------------------------
include(FetchContent)

# Create cmake directory if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/cmake)

# Include custom cmake modules
include(${CMAKE_SOURCE_DIR}/cmake/CompilerSettings.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Dependencies.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/SourceFiles.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/LinkLibraries.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/PokeRED.cmake)

# ------------------------------------------------------------------------------
# PPOkemon Library
# ------------------------------------------------------------------------------
add_library(ppokemon_libs STATIC ${GAMBATTE_SOURCES} ${TORCH_RL_SOURCES} ${ENVIRONMENT_SOURCES})

# Configure library
target_compile_definitions(ppokemon_libs PRIVATE ${COMMON_DEFINITIONS})
target_compile_options(ppokemon_libs PRIVATE ${COMMON_COMPILE_FLAGS})

# Set include directories and link libraries
set_project_includes(ppokemon_libs)
link_project_libraries(ppokemon_libs)

# Set specific flags for torch sources
foreach(source_file IN LISTS TORCH_RL_SOURCES)
  set_source_files_properties(${source_file} PROPERTIES
    COMPILE_FLAGS "-frtti -fexceptions"
  )
endforeach()

# ------------------------------------------------------------------------------
# Main Application (when implemented)
# ------------------------------------------------------------------------------
if(BUILD_MAIN_APP)
  # Check if main.cpp has content
  file(READ ${CMAKE_SOURCE_DIR}/src/main.cpp MAIN_CONTENT)
  string(STRIP "${MAIN_CONTENT}" MAIN_CONTENT)
  
  if(NOT "${MAIN_CONTENT}" STREQUAL "")
    add_executable(ppokemon ${DRL_SOURCES})
    
    # Configure executable
    target_compile_definitions(ppokemon PRIVATE ${COMMON_DEFINITIONS})
    target_compile_options(ppokemon PRIVATE ${COMMON_COMPILE_FLAGS})
    
    # Set include directories for the executable
    set_project_includes(ppokemon)
    
    # Link to the library and torch
    target_link_libraries(ppokemon PRIVATE ppokemon_libs)
    link_project_libraries(ppokemon)
    
    # Set specific flags for main sources
    foreach(source_file IN LISTS DRL_SOURCES)
      set_source_files_properties(${source_file} PROPERTIES
        COMPILE_FLAGS "-frtti -fexceptions"
      )
    endforeach()
    
    # Set RPATH for dynamic builds
    set_target_properties(ppokemon PROPERTIES
      BUILD_RPATH ${CMAKE_BINARY_DIR}
      INSTALL_RPATH ${CMAKE_BINARY_DIR}
    )
  else()
    message(STATUS "main.cpp is empty - skipping main application build")
  endif()
endif()

# ------------------------------------------------------------------------------
# Test Executable
# ------------------------------------------------------------------------------
if(BUILD_TEST_APP)
  add_executable(ppokemon_test ${CMAKE_SOURCE_DIR}/src/test.cpp)
  
  # Configure test executable
  target_compile_definitions(ppokemon_test PRIVATE ${COMMON_DEFINITIONS})
  target_compile_options(ppokemon_test PRIVATE ${COMMON_COMPILE_FLAGS})
  
  # Set include directories for the test executable
  set_project_includes(ppokemon_test)
  
  # Link to torch (test doesn't need the full library for now)
  link_project_libraries(ppokemon_test)
  
  # Set specific flags for test
  set_source_files_properties(${CMAKE_SOURCE_DIR}/src/test.cpp PROPERTIES
    COMPILE_FLAGS "-frtti -fexceptions"
  )
  
  # Set RPATH for dynamic builds
  set_target_properties(ppokemon_test PROPERTIES
    BUILD_RPATH ${CMAKE_BINARY_DIR}
    INSTALL_RPATH ${CMAKE_BINARY_DIR}
  )
  
  message(STATUS "Building test executable: ppokemon_test")
endif()

# Test executables are now defined in tests/CMakeLists.txt

# ------------------------------------------------------------------------------
# Tests subdirectory
# ------------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
