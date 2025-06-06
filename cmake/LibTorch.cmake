# LibTorch.cmake - LibTorch download and configuration

function(setup_libtorch)
  find_package(Torch QUIET
    HINTS
    ${CMAKE_SOURCE_DIR}/../libtorch
    ${CMAKE_SOURCE_DIR}/libtorch
    ${CMAKE_SOURCE_DIR}/../libtorch/lib
    ${CMAKE_SOURCE_DIR}/../libtorch/include
  )

  if(NOT Torch_FOUND)
    message(STATUS "Torch not found. Downloading libtorch...")
    
    # Determine download URL based on platform and options
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

    # Download and extract libtorch
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
      ${CMAKE_SOURCE_DIR}/libtorch/include
    )
  endif()

  if(Torch_FOUND)
    add_definitions(-DUSE_TORCH)
    message(STATUS "Found Torch: ${TORCH_LIBRARIES}")
    
    # Create a clean torch interface library
    if(NOT TARGET torch_clean)
      add_library(torch_clean INTERFACE)
      target_link_libraries(torch_clean INTERFACE ${TORCH_LIBRARIES})
      target_include_directories(torch_clean INTERFACE ${TORCH_INCLUDE_DIRS})
      target_link_directories(torch_clean INTERFACE ${TORCH_LIBRARY_DIRS})
      
      # Clean problematic compile flags
      get_target_property(TORCH_COMPILE_OPTIONS torch_clean INTERFACE_COMPILE_OPTIONS)
      if(TORCH_COMPILE_OPTIONS)
        list(REMOVE_ITEM TORCH_COMPILE_OPTIONS "-Wno-duplicate-decl-specifier")
        set_target_properties(torch_clean PROPERTIES INTERFACE_COMPILE_OPTIONS "${TORCH_COMPILE_OPTIONS}")
      endif()
    endif()
  else()
    message(FATAL_ERROR "Torch not found even after download attempt. Please check your configuration.")
  endif()
endfunction()

# Function to clean problematic compiler flags from torch targets
function(clean_torch_flags)
  # Clean all torch-related targets
  foreach(torch_target torch torch_cpu torch_cuda c10 c10_cuda caffe2_nvrtc torch_hip c10_hip)
    if(TARGET ${torch_target})
      get_target_property(interface_options ${torch_target} INTERFACE_COMPILE_OPTIONS)
      if(interface_options)
        list(REMOVE_ITEM interface_options "-Wno-duplicate-decl-specifier")
        set_target_properties(${torch_target} PROPERTIES INTERFACE_COMPILE_OPTIONS "${interface_options}")
      endif()
    endif()
  endforeach()

  # Remove problematic flags from global CMAKE flags
  string(REPLACE "-Wno-duplicate-decl-specifier" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-Wno-duplicate-decl-specifier" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  
  if(TORCH_CXX_FLAGS)
    string(REPLACE "-Wno-duplicate-decl-specifier" "" TORCH_CXX_FLAGS "${TORCH_CXX_FLAGS}")
  endif()
  
  # Propagate the cleaned flags back to parent scope
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" PARENT_SCOPE)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" PARENT_SCOPE)
endfunction()

# Call the setup function
setup_libtorch()
clean_torch_flags()
