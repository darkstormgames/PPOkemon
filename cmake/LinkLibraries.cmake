# LinkLibraries.cmake - Library linking configuration

# Function to link libraries to a target based on options
function(link_project_libraries target)
  # Link torch
  target_link_libraries(${target} PRIVATE torch_clean)

  # Link SDL2
  if(USE_SDL2)
    if(STATIC_LINK_LIBS)
      target_link_libraries(${target} PRIVATE SDL2-static)
    else()
      target_link_libraries(${target} PRIVATE SDL2::SDL2)
    endif()
  endif()

  # Link zlib and minizip
  if(STATIC_LINK_LIBS)
    target_link_libraries(${target} PRIVATE zlibstatic minizip)
  else()
    target_link_libraries(${target} PRIVATE ZLIB::ZLIB ${MINIZIP_LIB})
  endif()

  # Link OpenMP
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
  endif()
endfunction()

# Function to set include directories for a target
function(set_project_includes target)
  # Set the main include directories - this should cover most cases
  target_include_directories(${target} PRIVATE
    # Main include directory and subdirectories  
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/libgambatte
    ${CMAKE_SOURCE_DIR}/include/libgambatte/common
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
  )
  
  # Set public include directories for external usage
  target_include_directories(${target} PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/config
    ${CMAKE_SOURCE_DIR}/include/emulation
    ${CMAKE_SOURCE_DIR}/include/environments
    ${CMAKE_SOURCE_DIR}/include/torch
  )
  
  # Add torch includes if available
  if(TARGET torch_clean)
    target_include_directories(${target} PUBLIC ${TORCH_INCLUDE_DIRS})
  endif()
endfunction()
