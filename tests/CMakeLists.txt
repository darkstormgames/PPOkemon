# CMakeLists.txt for PPOkemon Tests
cmake_minimum_required(VERSION 3.18)

# Enable testing
enable_testing()

# Helper function to create a test target
function(add_ppokemon_test TEST_NAME SOURCE_FILE)
    # Create the test executable
    add_executable(${TEST_NAME} ${SOURCE_FILE})
    
    # Set C++ standard
    set_target_properties(${TEST_NAME} PROPERTIES CXX_STANDARD 17)
    
    # Set include directories (same as the main library)
    set_project_includes(${TEST_NAME})
    
    # Link against the PPOkemon library (contains all dependencies)
    target_link_libraries(${TEST_NAME} PRIVATE ppokemon_libs)
    link_project_libraries(${TEST_NAME})
    
    # Apply compile options
    target_compile_options(${TEST_NAME} PRIVATE -Wall -Wextra -O2)
    target_compile_definitions(${TEST_NAME} PRIVATE HAVE_STDINT_H)
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_definitions(${TEST_NAME} PRIVATE BUILD_DEBUG)
    endif()
    
    # Apply -frtti and -fexceptions for torch compatibility
    set_target_properties(${TEST_NAME} PROPERTIES
        COMPILE_FLAGS "-frtti -fexceptions"
    )
    
    # Set RPATH for dynamic builds
    set_target_properties(${TEST_NAME} PROPERTIES
        BUILD_RPATH ${CMAKE_BINARY_DIR}
        INSTALL_RPATH ${CMAKE_BINARY_DIR}
    )
    
    # Register as a test
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    
    # Set test properties
    set_tests_properties(${TEST_NAME} PROPERTIES
        TIMEOUT 300
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endfunction()

# Create test targets
add_ppokemon_test(test_networks test_networks.cpp)
add_ppokemon_test(test_environments test_environments.cpp)
add_ppokemon_test(test_utils test_utils.cpp)
add_ppokemon_test(test_advanced_utils test_advanced_utils.cpp)

# Additional test target that runs all tests
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --verbose
    DEPENDS test_networks test_environments test_utils test_advanced_utils
    COMMENT "Running all PPOkemon tests"
)

# Test target for continuous integration
add_custom_target(ci_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --timeout 60
    DEPENDS test_networks test_environments test_utils test_advanced_utils
    COMMENT "Running CI tests with timeout"
)

# Individual test runner targets for convenience
add_custom_target(run_network_tests
    COMMAND test_networks
    DEPENDS test_networks
    COMMENT "Running network tests"
)

add_custom_target(run_environment_tests
    COMMAND test_environments
    DEPENDS test_environments
    COMMENT "Running environment tests"
)

add_custom_target(run_utils_tests
    COMMAND test_utils
    DEPENDS test_utils
    COMMENT "Running utilities tests"
)

add_custom_target(run_advanced_utils_tests
    COMMAND test_advanced_utils
    DEPENDS test_advanced_utils
    COMMENT "Running advanced utilities tests"
)

# Memory check target (if valgrind is available)
find_program(VALGRIND_EXECUTABLE valgrind)
if(VALGRIND_EXECUTABLE)
    add_custom_target(memcheck_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -T memcheck
        DEPENDS test_networks test_environments
        COMMENT "Running tests with memory check"
    )
endif()

# Coverage target (if gcov is available)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_custom_target(coverage_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        COMMAND gcov -r ${CMAKE_CURRENT_BINARY_DIR}/*.gcno
        DEPENDS test_networks test_environments
        COMMENT "Running tests with coverage analysis"
    )
endif()

message(STATUS "Test targets configured:")
message(STATUS "  - test_networks: Network implementation tests")
message(STATUS "  - test_environments: Environment implementation tests")
message(STATUS "  - test_utils: Utilities implementation tests")
message(STATUS "  - test_advanced_utils: Advanced utilities implementation tests")
message(STATUS "  - run_all_tests: Run all tests with verbose output")
message(STATUS "  - ci_tests: Run tests for CI with timeout")
if(VALGRIND_EXECUTABLE)
    message(STATUS "  - memcheck_tests: Run tests with memory checking")
endif()