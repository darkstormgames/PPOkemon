# PokeRED.cmake - Pokemon Red/Blue ROM building configuration

# Function to setup RGBDS tools
function(setup_rgbds)
  set(RGBDS_VERSION "0.9.2")
  set(RGBDS_URL "https://github.com/gbdev/rgbds/releases/download/v${RGBDS_VERSION}/rgbds-${RGBDS_VERSION}-linux-x86_64.tar.xz")
  set(RGBDS_ARCHIVE "${CMAKE_BINARY_DIR}/rgbds-${RGBDS_VERSION}-linux-x86_64.tar.xz")
  
  # Check if RGBDS is already available locally
  if(EXISTS "${CMAKE_BINARY_DIR}/rgbasm")
    message(STATUS "Using local RGBDS installation")
    set(RGBDS_READY TRUE PARENT_SCOPE)
    set(RGBDS_BIN_DIR "${CMAKE_BINARY_DIR}" PARENT_SCOPE)
    return()
  endif()
  
  # Check if RGBDS is installed system-wide
  find_program(RGBASM_EXECUTABLE rgbasm)
  if(RGBASM_EXECUTABLE)
    message(STATUS "Using system RGBDS installation: ${RGBASM_EXECUTABLE}")
    set(RGBDS_READY TRUE PARENT_SCOPE)
    return()
  endif()
  
  message(STATUS "Setting up RGBDS ${RGBDS_VERSION}...")
  
  # Download RGBDS if not available
  if(NOT EXISTS ${RGBDS_ARCHIVE})
    message(STATUS "Downloading RGBDS...")
    file(DOWNLOAD ${RGBDS_URL} ${RGBDS_ARCHIVE} SHOW_PROGRESS STATUS DOWNLOAD_STATUS)
    
    list(GET DOWNLOAD_STATUS 0 DOWNLOAD_RESULT)
    if(NOT DOWNLOAD_RESULT EQUAL 0)
      message(WARNING "Failed to download RGBDS")
      set(RGBDS_READY FALSE PARENT_SCOPE)
      return()
    endif()
  endif()
  
  # Extract RGBDS tools
  message(STATUS "Extracting RGBDS...")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xf ${RGBDS_ARCHIVE}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    RESULT_VARIABLE EXTRACT_RESULT
  )
  
  if(EXTRACT_RESULT EQUAL 0)
    set(RGBDS_READY TRUE PARENT_SCOPE)
    set(RGBDS_BIN_DIR "${CMAKE_BINARY_DIR}" PARENT_SCOPE)
    message(STATUS "RGBDS setup complete")
  else()
    message(WARNING "Failed to extract RGBDS")
    set(RGBDS_READY FALSE PARENT_SCOPE)
  endif()
endfunction()

# Function to build Pokemon Red and Blue ROMs
function(build_pokemon_roms)
  if(NOT RGBDS_READY)
    message(STATUS "RGBDS not available. Skipping Pokemon ROM build.")
    return()
  endif()
  
  set(POKERED_DIR "${CMAKE_BINARY_DIR}/pokered")
  
  # Check if ROMs already exist
  if(EXISTS "${CMAKE_SOURCE_DIR}/bin/pokered.gbc" AND EXISTS "${CMAKE_SOURCE_DIR}/bin/pokeblue.gbc")
    message(STATUS "Pokemon Red and Blue ROMs already exist")
    set(POKEMON_ROMS_AVAILABLE TRUE PARENT_SCOPE)
    return()
  endif()
  
  # Clone Pokemon Red repository if not exists
  if(NOT EXISTS ${POKERED_DIR})
    message(STATUS "Cloning Pokemon Red repository...")
    find_program(GIT_EXECUTABLE git)
    if(NOT GIT_EXECUTABLE)
      message(WARNING "Git not found. Cannot clone Pokemon repository.")
      return()
    endif()
    
    execute_process(
      COMMAND ${GIT_EXECUTABLE} clone https://github.com/pret/pokered.git
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      RESULT_VARIABLE CLONE_RESULT
    )
    
    if(NOT CLONE_RESULT EQUAL 0)
      message(WARNING "Failed to clone Pokemon repository.")
      return()
    endif()
  endif()
  
  # Prepare environment for building
  if(RGBDS_BIN_DIR)
    set(BUILD_ENV "PATH=${RGBDS_BIN_DIR}:$ENV{PATH}")
  endif()
  
  # Build both ROMs in one command for efficiency
  message(STATUS "Building Pokemon Red and Blue ROMs...")
  if(RGBDS_BIN_DIR)
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E env ${BUILD_ENV} make red blue
      WORKING_DIRECTORY ${POKERED_DIR}
      RESULT_VARIABLE BUILD_RESULT
    )
  else()
    execute_process(
      COMMAND make red blue
      WORKING_DIRECTORY ${POKERED_DIR}
      RESULT_VARIABLE BUILD_RESULT
    )
  endif()
  
  # Copy ROMs to bin directory
  file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
  
  set(ROMS_BUILT FALSE)
  if(BUILD_RESULT EQUAL 0)
    if(EXISTS "${POKERED_DIR}/pokered.gbc")
      file(COPY "${POKERED_DIR}/pokered.gbc" DESTINATION "${CMAKE_SOURCE_DIR}/bin/")
      message(STATUS "Pokemon Red ROM built successfully")
      set(ROMS_BUILT TRUE)
    endif()
    
    if(EXISTS "${POKERED_DIR}/pokeblue.gbc")
      file(COPY "${POKERED_DIR}/pokeblue.gbc" DESTINATION "${CMAKE_SOURCE_DIR}/bin/")
      message(STATUS "Pokemon Blue ROM built successfully")
      set(ROMS_BUILT TRUE)
    endif()
  endif()
  
  if(ROMS_BUILT)
    set(POKEMON_ROMS_AVAILABLE TRUE PARENT_SCOPE)
  else()
    message(WARNING "Failed to build Pokemon ROMs")
  endif()
endfunction()

# Function to create custom build targets
function(add_pokemon_rom_targets)
  if(NOT RGBDS_READY)
    return()
  endif()
  
  set(POKERED_DIR "${CMAKE_BINARY_DIR}/pokered")
  
  # Create custom target for building both ROMs
  if(RGBDS_BIN_DIR)
    add_custom_target(build_pokemon_roms
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/bin
      COMMAND ${CMAKE_COMMAND} -E env PATH=${RGBDS_BIN_DIR}:$ENV{PATH} make red blue
      COMMAND ${CMAKE_COMMAND} -E copy_if_different pokered.gbc ${CMAKE_SOURCE_DIR}/bin/ || true
      COMMAND ${CMAKE_COMMAND} -E copy_if_different pokeblue.gbc ${CMAKE_SOURCE_DIR}/bin/ || true
      WORKING_DIRECTORY ${POKERED_DIR}
      COMMENT "Building Pokemon Red and Blue ROMs"
      VERBATIM
    )
  else()
    add_custom_target(build_pokemon_roms
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/bin
      COMMAND make red blue
      COMMAND ${CMAKE_COMMAND} -E copy_if_different pokered.gbc ${CMAKE_SOURCE_DIR}/bin/ || true
      COMMAND ${CMAKE_COMMAND} -E copy_if_different pokeblue.gbc ${CMAKE_SOURCE_DIR}/bin/ || true
      WORKING_DIRECTORY ${POKERED_DIR}
      COMMENT "Building Pokemon Red and Blue ROMs"
      VERBATIM
    )
  endif()
  
  # Create clean target
  add_custom_target(clean_pokemon_roms
    COMMAND make clean
    WORKING_DIRECTORY ${POKERED_DIR}
    COMMENT "Cleaning Pokemon ROM build files"
    VERBATIM
  )
  
  # Legacy target for compatibility
  add_custom_target(build_pokered_rom
    DEPENDS build_pokemon_roms
    COMMENT "Legacy target - use build_pokemon_roms instead"
  )
endfunction()

# Main setup function
function(setup_pokemon_rom)
  message(STATUS "Setting up Pokemon ROM build environment...")
  
  setup_rgbds()
  
  if(RGBDS_READY)
    build_pokemon_roms()
    add_pokemon_rom_targets()
    
    if(POKEMON_ROMS_AVAILABLE)
      message(STATUS "Pokemon ROMs are ready for testing")
      set(POKERED_ROM_AVAILABLE TRUE PARENT_SCOPE)
      add_compile_definitions(POKERED_ROM_AVAILABLE)
    else()
      message(STATUS "Pokemon ROM build incomplete. Use 'make build_pokemon_roms' to build manually")
    endif()
  else()
    message(STATUS "RGBDS setup failed. Pokemon ROMs will not be available.")
  endif()
endfunction()

# Call setup if enabled
if(BUILD_POKERED_ROM)
  setup_pokemon_rom()
endif()
