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
    
    # Apply compile options and definitions (use the same as main project)
    target_compile_definitions(${TEST_NAME} PRIVATE ${COMMON_DEFINITIONS})
    target_compile_options(${TEST_NAME} PRIVATE ${COMMON_COMPILE_FLAGS})
    
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

# Network tests
add_ppokemon_test(test_networks test_networks.cpp)
add_ppokemon_test(test_networks_mlp test_networks_mlp.cpp)
add_ppokemon_test(test_networks_cnn test_networks_cnn.cpp)
add_ppokemon_test(test_networks_a2c test_networks_a2c.cpp)
add_ppokemon_test(test_networks_a3c test_networks_a3c.cpp)
add_ppokemon_test(test_networks_integration test_networks_integration.cpp)

# Environment tests
add_ppokemon_test(test_environments test_environments.cpp)
add_ppokemon_test(test_environments_base test_environments_base.cpp)
add_ppokemon_test(test_environments_vectorized test_environments_vectorized.cpp)
add_ppokemon_test(test_environments_running_stats test_environments_running_stats.cpp)
add_ppokemon_test(test_pokered_rewards test_pokered_rewards.cpp)

# Utility tests
add_ppokemon_test(test_utils test_utils.cpp)
add_ppokemon_test(test_utils_seed test_utils_seed.cpp)
add_ppokemon_test(test_utils_replay test_utils_replay.cpp)
add_ppokemon_test(test_utils_profiler test_utils_profiler.cpp)
add_ppokemon_test(test_utils_scheduler test_utils_scheduler.cpp)
add_ppokemon_test(test_utils_ml test_utils_ml.cpp)

# Algorithm tests
add_ppokemon_test(test_algorithms_ppo_simple test_algorithms_ppo_simple.cpp)
add_ppokemon_test(test_algorithms_ppo_comprehensive test_algorithms_ppo_comprehensive.cpp)

# Training tests
add_ppokemon_test(test_training test_training.cpp)
add_ppokemon_test(test_training_metrics test_training_metrics.cpp)
add_ppokemon_test(test_training_evaluator test_training_evaluator.cpp)
add_ppokemon_test(test_training_callbacks test_training_callbacks.cpp)
add_ppokemon_test(test_training_trainer test_training_trainer.cpp)

# Legacy advanced utils test (if needed)
add_ppokemon_test(test_advanced_utils test_advanced_utils.cpp)

# Additional test target that runs all tests
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --verbose
    DEPENDS 
        test_networks test_networks_mlp test_networks_cnn test_networks_a2c test_networks_a3c test_networks_integration
        test_environments test_environments_base test_environments_vectorized test_environments_running_stats
        test_utils test_utils_seed test_utils_replay test_utils_profiler test_utils_scheduler test_utils_ml
        test_algorithms_ppo
        test_training test_training_metrics test_training_evaluator test_training_callbacks test_training_trainer
        test_advanced_utils
    COMMENT "Running all PPOkemon tests"
)

# Test target for continuous integration
add_custom_target(ci_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --timeout 60
    DEPENDS 
        test_networks test_networks_mlp test_networks_cnn test_networks_a2c test_networks_a3c
        test_environments test_environments_base test_environments_vectorized test_environments_running_stats
        test_utils test_utils_seed test_utils_replay test_utils_profiler test_utils_scheduler test_utils_ml
        test_algorithms_ppo
        test_training test_training_metrics test_training_evaluator test_training_callbacks test_training_trainer
    COMMENT "Running CI tests with timeout"
)

# Individual test runner targets for convenience
add_custom_target(run_network_tests
    COMMAND test_networks
    DEPENDS test_networks
    COMMENT "Running network tests"
)

add_custom_target(run_network_tests_detailed
    COMMAND test_networks_mlp && test_networks_cnn && test_networks_a2c && test_networks_a3c && test_networks_integration
    DEPENDS test_networks_mlp test_networks_cnn test_networks_a2c test_networks_a3c test_networks_integration
    COMMENT "Running detailed network tests"
)

add_custom_target(run_algorithm_tests
    COMMAND test_algorithms_ppo
    DEPENDS test_algorithms_ppo
    COMMENT "Running algorithm tests"
)

add_custom_target(run_environment_tests
    COMMAND test_environments
    DEPENDS test_environments
    COMMENT "Running environment tests"
)

add_custom_target(run_environment_tests_detailed
    COMMAND test_environments_base && test_environments_vectorized && test_environments_running_stats
    DEPENDS test_environments_base test_environments_vectorized test_environments_running_stats
    COMMENT "Running detailed environment tests"
)

add_custom_target(run_utils_tests
    COMMAND test_utils
    DEPENDS test_utils
    COMMENT "Running utilities tests"
)

add_custom_target(run_utils_tests_detailed
    COMMAND test_utils_seed && test_utils_replay && test_utils_profiler && test_utils_scheduler && test_utils_ml
    DEPENDS test_utils_seed test_utils_replay test_utils_profiler test_utils_scheduler test_utils_ml
    COMMENT "Running detailed utility tests"
)

add_custom_target(run_training_tests
    COMMAND test_training
    DEPENDS test_training
    COMMENT "Running training component tests"
)

add_custom_target(run_training_tests_detailed
    COMMAND test_training_metrics && test_training_evaluator && test_training_callbacks && test_training_trainer
    DEPENDS test_training_metrics test_training_evaluator test_training_callbacks test_training_trainer
    COMMENT "Running detailed training tests"
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
message(STATUS "  Network Tests:")
message(STATUS "    - test_networks: Main network implementation tests")
message(STATUS "    - test_networks_mlp: Multi-layer perceptron tests")
message(STATUS "    - test_networks_cnn: Convolutional neural network tests")
message(STATUS "    - test_networks_a2c: A2C network architecture tests")
message(STATUS "    - test_networks_a3c: A3C network architecture tests")
message(STATUS "    - test_networks_integration: Network integration tests")
message(STATUS "  Environment Tests:")
message(STATUS "    - test_environments: Main environment implementation tests")
message(STATUS "    - test_environments_base: Base environment tests")
message(STATUS "    - test_environments_vectorized: Vectorized environment tests")
message(STATUS "    - test_environments_running_stats: Running statistics tests")
message(STATUS "  Utility Tests:")
message(STATUS "    - test_utils: Main utilities implementation tests")
message(STATUS "    - test_utils_seed: Seed manager tests")
message(STATUS "    - test_utils_replay: Replay buffer tests")
message(STATUS "    - test_utils_profiler: Profiler tests")
message(STATUS "    - test_utils_scheduler: Learning rate scheduler tests")
message(STATUS "    - test_utils_ml: ML utilities tests")
message(STATUS "  Training Tests:")
message(STATUS "    - test_training: Main training implementation tests")
message(STATUS "    - test_training_metrics: Training metrics tests")
message(STATUS "    - test_training_evaluator: Training evaluator tests")
message(STATUS "    - test_training_callbacks: Training callbacks tests")
message(STATUS "    - test_training_trainer: Trainer base class tests")
message(STATUS "  Meta Targets:")
message(STATUS "    - run_all_tests: Run all tests with verbose output")
message(STATUS "    - ci_tests: Run tests for CI with timeout")
message(STATUS "    - run_*_tests_detailed: Run detailed modular tests for each category")
if(VALGRIND_EXECUTABLE)
    message(STATUS "    - memcheck_tests: Run tests with memory checking")
endif()