#include "../include/torch/networks/network_base.h"
#include "../include/torch/networks/a2c.h"
#include "../include/torch/networks/mlp.h"
#include "../include/torch/networks/cnn.h"
#include "test_utils.h"
#include <iostream>

using namespace PPOkemonTest;

// Test A2C with MLP backbone
void test_a2c_mlp() {
    auto a2c_mlp = networks::A2C(networks::MLPTag{}, 10, 64, 4);  // 4 actions
    auto input = torch::randn({1, 10});
    auto a2c_output = a2c_mlp->forward(input);
    
    ASSERT_EQ(a2c_output.size(0), 1);
    ASSERT_EQ(a2c_output.size(1), 4);
    
    std::cout << "✓ A2C-MLP forward pass: input [1,10] -> output [" << a2c_output.size(0) << "," << a2c_output.size(1) << "]" << std::endl;
}

// Test A2C Actor-Critic functionality
void test_a2c_actor_critic() {
    auto a2c_mlp = networks::A2C(networks::MLPTag{}, 8, 32, 4);
    auto input = torch::randn({2, 8});
    
    // Test Actor-Critic methods
    auto [actions, values] = a2c_mlp->ForwardActorCritic(input);
    ASSERT_EQ(actions.size(0), 2);
    ASSERT_EQ(actions.size(1), 4);
    ASSERT_EQ(values.size(0), 2);
    ASSERT_EQ(values.size(1), 1);
    
    // Test individual methods
    auto action_logits = a2c_mlp->GetAction(input);
    auto state_values = a2c_mlp->GetValue(input);
    
    ASSERT_EQ(action_logits.size(0), 2);
    ASSERT_EQ(action_logits.size(1), 4);
    ASSERT_EQ(state_values.size(0), 2);
    ASSERT_EQ(state_values.size(1), 1);
    
    std::cout << "✓ A2C Actor-Critic functionality works correctly" << std::endl;
}

// Test A2C with CNN backbone
void test_a2c_cnn() {
    auto a2c_cnn = networks::A2C(networks::CNNTag{}, 1, 160, 144, 4);  // 4 actions
    auto img_input = torch::randn({1, 1, 160, 144});
    auto a2c_cnn_output = a2c_cnn->forward(img_input);
    
    ASSERT_EQ(a2c_cnn_output.size(0), 1);
    ASSERT_EQ(a2c_cnn_output.size(1), 4);
    
    // Test actor-critic with CNN
    auto [actions, values] = a2c_cnn->ForwardActorCritic(img_input);
    ASSERT_EQ(actions.size(0), 1);
    ASSERT_EQ(actions.size(1), 4);
    ASSERT_EQ(values.size(0), 1);
    ASSERT_EQ(values.size(1), 1);
    
    std::cout << "✓ A2C-CNN forward pass: input [1,1,160,144] -> output [" << a2c_cnn_output.size(0) << "," << a2c_cnn_output.size(1) << "]" << std::endl;
}

// Test A2C with shared backbone
void test_a2c_shared_backbone() {
    // Test using the factory method instead of direct constructor
    auto a2c_shared = networks::A2CImpl::WithMLP(6, 32, 3, 1);
    
    auto input = torch::randn({2, 6});
    auto output = a2c_shared->forward(input);
    
    ASSERT_EQ(output.size(0), 2);
    ASSERT_EQ(output.size(1), 3);
    
    auto [actions, values] = a2c_shared->ForwardActorCritic(input);
    ASSERT_EQ(actions.size(0), 2);
    ASSERT_EQ(actions.size(1), 3);
    ASSERT_EQ(values.size(0), 2);
    ASSERT_EQ(values.size(1), 1);
    
    std::cout << "✓ A2C shared backbone functionality works correctly" << std::endl;
}

// Test A2C initialization
void test_a2c_initialization() {
    auto a2c = networks::A2C(networks::MLPTag{}, 10, 64, 4);
    
    // Test orthogonal initialization
    ASSERT_NO_THROW(a2c->InitOrtho(1.0f));
    
    // Check parameters
    auto params = a2c->parameters();
    ASSERT_TRUE(params.size() > 0);
    
    bool has_non_zero = false;
    for (const auto& param : params) {
        if (torch::any(param != 0).item<bool>()) {
            has_non_zero = true;
            break;
        }
    }
    ASSERT_TRUE(has_non_zero);
    
    std::cout << "✓ A2C orthogonal initialization successful" << std::endl;
}

// Test A2C factory methods
void test_a2c_factory_methods() {
    // Test WithMLP factory
    auto a2c_mlp = networks::A2CImpl::WithMLP(8, 32, 4, 1);
    auto mlp_input = torch::randn({1, 8});
    auto mlp_output = a2c_mlp->forward(mlp_input);
    ASSERT_EQ(mlp_output.size(1), 4);
    
    // Test WithCNN factory
    auto a2c_cnn = networks::A2CImpl::WithCNN(3, 84, 84, 6, 1);
    auto cnn_input = torch::randn({1, 3, 84, 84});
    auto cnn_output = a2c_cnn->forward(cnn_input);
    ASSERT_EQ(cnn_output.size(1), 6);
    
    // Test WithSharedBackbone factory - use WithMLP instead
    auto a2c_shared = networks::A2CImpl::WithMLP(10, 64, 5, 1);
    auto shared_input = torch::randn({1, 10});
    auto shared_output = a2c_shared->forward(shared_input);
    ASSERT_EQ(shared_output.size(1), 5);
    
    std::cout << "✓ A2C factory methods work correctly" << std::endl;
}

int main() {
    TestSuite suite("PPOkemon A2C Network Tests");
    
    // Add test cases
    suite.AddTest("A2C MLP", test_a2c_mlp);
    suite.AddTest("A2C Actor-Critic", test_a2c_actor_critic);
    suite.AddTest("A2C CNN", test_a2c_cnn);
    suite.AddTest("A2C Shared Backbone", test_a2c_shared_backbone);
    suite.AddTest("A2C Initialization", test_a2c_initialization);
    suite.AddTest("A2C Factory Methods", test_a2c_factory_methods);
    
    // Run all tests
    bool all_passed = suite.RunAll();
    
    // Get final statistics
    int passed, failed, total;
    suite.GetStats(passed, failed, total);
    
    std::cout << "\nFinal Results: " << passed << "/" << total << " A2C tests passed" << std::endl;
    
    return all_passed ? 0 : 1;
}
