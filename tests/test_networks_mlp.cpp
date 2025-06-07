#include "../include/torch/networks/network_base.h"
#include "../include/torch/networks/mlp.h"
#include "test_utils.h"
#include <iostream>

using namespace PPOkemonTest;

// Test MLP Network basic functionality
void test_mlp_basic() {
    auto mlp = networks::MLP(10, 64, 32);
    auto input = torch::randn({1, 10});
    auto output = mlp->forward(input);
    
    ASSERT_EQ(output.size(0), 1);
    ASSERT_EQ(output.size(1), 32);
    
    std::cout << "✓ MLP forward pass: input [1,10] -> output [" << output.size(0) << "," << output.size(1) << "]" << std::endl;
}

// Test MLP orthogonal initialization
void test_mlp_initialization() {
    auto mlp = networks::MLP(8, 32, 16);
    
    // Test orthogonal initialization
    ASSERT_NO_THROW(mlp->InitOrtho(1.0f));
    
    // Check that weights are not all zeros after initialization
    auto params = mlp->parameters();
    bool has_non_zero = false;
    for (const auto& param : params) {
        if (torch::any(param != 0).item<bool>()) {
            has_non_zero = true;
            break;
        }
    }
    ASSERT_TRUE(has_non_zero);
    
    std::cout << "✓ MLP orthogonal initialization successful" << std::endl;
}

// Test MLP with different sizes
void test_mlp_various_sizes() {
    // Test small network
    auto small_mlp = networks::MLP(4, 16, 2);
    auto small_input = torch::randn({2, 4});
    auto small_output = small_mlp->forward(small_input);
    ASSERT_EQ(small_output.size(0), 2);
    ASSERT_EQ(small_output.size(1), 2);
    
    // Test larger network
    auto large_mlp = networks::MLP(128, 256, 64);
    auto large_input = torch::randn({4, 128});
    auto large_output = large_mlp->forward(large_input);
    ASSERT_EQ(large_output.size(0), 4);
    ASSERT_EQ(large_output.size(1), 64);
    
    std::cout << "✓ MLP various sizes work correctly" << std::endl;
}

// Test MLP parameter count
void test_mlp_parameters() {
    auto mlp = networks::MLP(10, 64, 32);
    auto params = mlp->parameters();
    
    ASSERT_TRUE(params.size() > 0);
    
    // Count total parameters
    int64_t total_params = 0;
    for (const auto& param : params) {
        total_params += param.numel();
    }
    
    // Expected: (10*64 + 64) + (64*64 + 64) + (64*32 + 32) = 640 + 64 + 4096 + 64 + 2048 + 32 = 6944
    int64_t expected_params = (10 * 64 + 64) + (64 * 64 + 64) + (64 * 32 + 32);
    ASSERT_EQ(total_params, expected_params);
    
    std::cout << "✓ MLP parameter count: " << total_params << " (expected: " << expected_params << ")" << std::endl;
}

int main() {
    TestSuite suite("PPOkemon MLP Network Tests");
    
    // Add test cases
    suite.AddTest("MLP Basic Functionality", test_mlp_basic);
    suite.AddTest("MLP Initialization", test_mlp_initialization);
    suite.AddTest("MLP Various Sizes", test_mlp_various_sizes);
    suite.AddTest("MLP Parameters", test_mlp_parameters);
    
    // Run all tests
    bool all_passed = suite.RunAll();
    
    // Get final statistics
    int passed, failed, total;
    suite.GetStats(passed, failed, total);
    
    std::cout << "\nFinal Results: " << passed << "/" << total << " MLP tests passed" << std::endl;
    
    return all_passed ? 0 : 1;
}
