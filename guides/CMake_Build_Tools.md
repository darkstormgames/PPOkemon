No, you cannot have multiple CMakeLists.txt files in the same folder. CMake expects exactly one CMakeLists.txt file per directory. However, there are several ways to achieve what you want:

## 1. **Use Different Build Configurations**

Keep one CMakeLists.txt but use options to build different targets:

````cmake
cmake_minimum_required(VERSION 3.18)
project(PPOkemon)

# Options for different build targets
option(BUILD_MAIN_APP "Build the main PPOkemon application" ON)
option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_TOOLS "Build utility tools" OFF)

# Common setup
set(CMAKE_CXX_STANDARD 17)
# ...existing common configuration...

# Main application
if(BUILD_MAIN_APP)
    add_executable(drl_test ${DRL_SOURCES} ${GAMBATTE_SOURCES} ${TORCH_RL_SOURCES})
    # ...existing drl_test configuration...
endif()

# Unit tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Tools
if(BUILD_TOOLS)
    add_subdirectory(tools)
endif()
````

Then build different configurations:
```bash
# Build main app only
cmake -B build -DBUILD_TESTS=OFF -DBUILD_TOOLS=OFF

# Build tests only
cmake -B build_tests -DBUILD_MAIN_APP=OFF -DBUILD_TESTS=ON

# Build everything
cmake -B build_all -DBUILD_TESTS=ON -DBUILD_TOOLS=ON
```

## 2. **Use Separate Subdirectories**

Create subdirectories for different projects:

```
PPOkemon/
├── CMakeLists.txt          # Root CMake file
├── app/
│   └── CMakeLists.txt      # Main application
├── tests/
│   └── CMakeLists.txt      # Unit tests
└── tools/
    └── CMakeLists.txt      # Utility tools
```

Root CMakeLists.txt:
````cmake
cmake_minimum_required(VERSION 3.18)
project(PPOkemon)

# Common configuration
set(CMAKE_CXX_STANDARD 17)
# ...common setup...

# Options
option(BUILD_APP "Build main application" ON)
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_TOOLS "Build tools" OFF)

# Add subdirectories conditionally
if(BUILD_APP)
    add_subdirectory(app)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_TOOLS)
    add_subdirectory(tools)
endif()
````

## 3. **Use CMake Presets** (Recommended)

Create a `CMakePresets.json` file:

````json
{
  "version": 3,
  "configurePresets": [
    {
      "name": "default",
      "hidden": true,
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/${presetName}",
      "cacheVariables": {
        "CMAKE_CXX_STANDARD": "17"
      }
    },
    {
      "name": "main-app",
      "inherits": "default",
      "displayName": "Main Application",
      "cacheVariables": {
        "BUILD_MAIN_APP": "ON",
        "BUILD_TESTS": "OFF",
        "USE_SDL2": "ON"
      }
    },
    {
      "name": "tests",
      "inherits": "default",
      "displayName": "Unit Tests",
      "cacheVariables": {
        "BUILD_MAIN_APP": "OFF",
        "BUILD_TESTS": "ON"
      }
    },
    {
      "name": "all",
      "inherits": "default",
      "displayName": "All Targets",
      "cacheVariables": {
        "BUILD_MAIN_APP": "ON",
        "BUILD_TESTS": "ON",
        "BUILD_TOOLS": "ON"
      }
    }
  ],
  "buildPresets": [
    {
      "name": "main-app",
      "configurePreset": "main-app"
    },
    {
      "name": "tests",
      "configurePreset": "tests"
    }
  ],
  "testPresets": [
    {
      "name": "tests",
      "configurePreset": "tests",
      "output": {
        "outputOnFailure": true
      }
    }
  ]
}
````

Then use:
```bash
# Configure and build main app
cmake --preset main-app
cmake --build --preset main-app

# Configure and build tests
cmake --preset tests
cmake --build --preset tests
ctest --preset tests
```

## 4. **Include Additional CMake Files**

You can include other `.cmake` files:

````cmake
cmake_minimum_required(VERSION 3.18)
project(PPOkemon)

# Include different configurations
if(BUILD_TYPE STREQUAL "TESTS")
    include(cmake/tests.cmake)
elseif(BUILD_TYPE STREQUAL "TOOLS")
    include(cmake/tools.cmake)
else()
    include(cmake/main_app.cmake)
endif()
````

````cmake
# Test-specific configuration
enable_testing()
include(FetchContent)

# Fetch Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

# Add test executable
add_executable(ppokemon_tests
    tests/test_main.cpp
    # ... test files
)

target_link_libraries(ppokemon_tests
    gtest_main
    gmock
    ${TORCH_LIBRARIES}
)
````

## 5. **For Your PPOkemon Project**

I recommend modifying your existing CMakeLists.txt to support multiple targets:

````cmake
# ...existing code...

# Add option for tests at the top
option(BUILD_TESTS "Build unit tests" OFF)

# ...existing code...

# After your main executable configuration, add:
if(BUILD_TESTS)
    # Download and configure Google Test
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    FetchContent_MakeAvailable(googletest)
    
    enable_testing()
    
    # Create test executable
    add_executable(ppokemon_tests
        tests/test_main.cpp
        tests/game/test_Pokemon.cpp
        # Add more test files here
    )
    
    # Include directories for tests
    target_include_directories(ppokemon_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )
    
    # Link test libraries
    target_link_libraries(ppokemon_tests PRIVATE
        gtest_main
        gmock
        torch_clean  # Reuse your cleaned torch library
        ${GAMBATTE_SOURCES}  # If needed for tests
    )
    
    # Discover tests
    include(GoogleTest)
    gtest_discover_tests(ppokemon_tests)
endif()
````

This way, you maintain a single CMakeLists.txt but can build different targets based on your needs.