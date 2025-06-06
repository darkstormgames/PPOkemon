#include "../include/torch/networks/network_base.h"
#include "../include/torch/networks/mlp.h"
#include "../include/torch/networks/cnn.h"
#include "../include/torch/networks/a2c.h"
#include "../include/torch/networks/a3c.h"
#include "test_utils.h"
#include <iostream>

using namespace PPOkemonTest;

// Test MLP Network functionality
void test_mlp_network() {
    auto mlp = networks::MLP(10, 64, 32);
    auto input = torch::randn({1, 10});
    auto output = mlp->forward(input);
    
    ASSERT_EQ(output.size(0), 1);
    ASSERT_EQ(output.size(1), 32);
    
    // Test orthogonal initialization
    ASSERT_NO_THROW(mlp->InitOrtho(1.0f));
}

// Test CNN Network functionality  
void test_cnn_network() {
    auto cnn = networks::CNNBody(1, 160, 144);  // Gameboy screen
    auto img_input = torch::randn({1, 1, 160, 144});
    auto cnn_output = cnn->forward(img_input);
    
    ASSERT_EQ(cnn_output.size(0), 1);
    ASSERT_TRUE(cnn_output.size(1) > 0);  // Output should have some features
}

// Test A2C Network functionality
void test_a2c_networks() {
    // Test A2C with MLP
    auto a2c_mlp = networks::A2C(networks::MLPTag{}, 10, 64, 4);  // 4 actions
    auto input = torch::randn({1, 10});
    auto a2c_output = a2c_mlp->forward(input);
    
    ASSERT_EQ(a2c_output.size(0), 1);
    ASSERT_EQ(a2c_output.size(1), 4);
    
    // Test Actor-Critic methods
    auto [actions, values] = a2c_mlp->ForwardActorCritic(input);
    ASSERT_EQ(actions.size(0), 1);
    ASSERT_EQ(actions.size(1), 4);
    ASSERT_EQ(values.size(0), 1);
    ASSERT_EQ(values.size(1), 1);
    
    // Test orthogonal initialization
    ASSERT_NO_THROW(a2c_mlp->InitOrtho(1.0f));
    
    // Test A2C with CNN
    auto a2c_cnn = networks::A2C(networks::CNNTag{}, 1, 160, 144, 4);  // 4 actions
    auto img_input = torch::randn({1, 1, 160, 144});
    auto a2c_cnn_output = a2c_cnn->forward(img_input);
    
    ASSERT_EQ(a2c_cnn_output.size(0), 1);
    ASSERT_EQ(a2c_cnn_output.size(1), 4);
}

// Test A3C Network functionality
void test_a3c_networks() {
    auto a3c_mlp = networks::A3C(networks::MLPTag{}, 10, 64, 4);  // 4 actions
    auto input = torch::randn({1, 10});
    auto a3c_output = a3c_mlp->forward(input);
    
    ASSERT_EQ(a3c_output.size(0), 1);
    ASSERT_EQ(a3c_output.size(1), 4);
    
    // Test orthogonal initialization
    ASSERT_NO_THROW(a3c_mlp->InitOrtho(1.0f));
}

// Test network parameter consistency
void test_network_parameters() {
    auto mlp = networks::MLP(10, 64, 32);
    auto a2c_mlp = networks::A2C(networks::MLPTag{}, 10, 64, 4);
    
    // Verify that networks have parameters
    auto mlp_params = mlp->parameters();
    auto a2c_params = a2c_mlp->parameters();
    
    ASSERT_TRUE(mlp_params.size() > 0);
    ASSERT_TRUE(a2c_params.size() > 0);
    
    // Test that parameters are properly initialized (not all zeros)
    bool has_non_zero = false;
    for (const auto& param : mlp_params) {
        if (torch::any(param != 0).item<bool>()) {
            has_non_zero = true;
            break;
        }
    }
    ASSERT_TRUE(has_non_zero);
}

int main() {
    TestSuite suite("PPOkemon Network Tests");
    
    // Add test cases
    suite.AddTest("MLP Network", test_mlp_network);
    suite.AddTest("CNN Network", test_cnn_network);
    suite.AddTest("A2C Networks", test_a2c_networks);
    suite.AddTest("A3C Networks", test_a3c_networks);
    suite.AddTest("Network Parameters", test_network_parameters);
    
    // Run all tests
    bool all_passed = suite.RunAll();
    
    // Get final statistics
    int passed, failed, total;
    suite.GetStats(passed, failed, total);
    
    std::cout << "\nFinal Results: " << passed << "/" << total << " tests passed" << std::endl;
    
    return all_passed ? 0 : 1;
}
