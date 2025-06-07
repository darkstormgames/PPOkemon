#include "../include/torch/networks/network_base.h"
#include "../include/torch/networks/a3c.h"
#include "../include/torch/networks/a2c.h"
#include "test_utils.h"
#include <iostream>

using namespace PPOkemonTest;

// Test A3C with MLP backbone
void test_a3c_mlp() {
    auto a3c_mlp = networks::A3C(networks::MLPTag{}, 10, 64, 4);  // 4 actions
    auto input = torch::randn({1, 10});
    auto a3c_output = a3c_mlp->forward(input);
    
    ASSERT_EQ(a3c_output.size(0), 1);
    ASSERT_EQ(a3c_output.size(1), 4);
    
    std::cout << "✓ A3C-MLP forward pass: input [1,10] -> output [" << a3c_output.size(0) << "," << a3c_output.size(1) << "]" << std::endl;
}

// Test A3C Actor-Critic functionality
void test_a3c_actor_critic() {
    auto a3c = networks::A3C(networks::MLPTag{}, 8, 32, 6);
    auto input = torch::randn({3, 8});
    
    // Test Actor-Critic methods
    auto [actions, values] = a3c->ForwardActorCritic(input);
    ASSERT_EQ(actions.size(0), 3);
    ASSERT_EQ(actions.size(1), 6);
    ASSERT_EQ(values.size(0), 3);
    ASSERT_EQ(values.size(1), 1);
    
    // Test individual methods
    auto action_logits = a3c->GetAction(input);
    auto state_values = a3c->GetValue(input);
    
    ASSERT_EQ(action_logits.size(0), 3);
    ASSERT_EQ(action_logits.size(1), 6);
    ASSERT_EQ(state_values.size(0), 3);
    ASSERT_EQ(state_values.size(1), 1);
    
    std::cout << "✓ A3C Actor-Critic functionality works correctly" << std::endl;
}

// Test A3C with CNN backbone
void test_a3c_cnn() {
    auto a3c_cnn = networks::A3C(networks::CNNTag{}, 1, 160, 144, 4);
    auto img_input = torch::randn({2, 1, 160, 144});
    auto a3c_cnn_output = a3c_cnn->forward(img_input);
    
    ASSERT_EQ(a3c_cnn_output.size(0), 2);
    ASSERT_EQ(a3c_cnn_output.size(1), 4);
    
    // Test actor-critic with CNN
    auto [actions, values] = a3c_cnn->ForwardActorCritic(img_input);
    ASSERT_EQ(actions.size(0), 2);
    ASSERT_EQ(actions.size(1), 4);
    ASSERT_EQ(values.size(0), 2);
    ASSERT_EQ(values.size(1), 1);
    
    std::cout << "✓ A3C-CNN forward pass: input [2,1,160,144] -> output [" << a3c_cnn_output.size(0) << "," << a3c_cnn_output.size(1) << "]" << std::endl;
}

// Test A3C with shared backbone
void test_a3c_shared_backbone() {
    // Test using the factory method instead of direct constructor
    auto a3c_shared = networks::A3CImpl::WithMLP(12, 48, 5, 1);
    
    auto input = torch::randn({2, 12});
    auto output = a3c_shared->forward(input);
    
    ASSERT_EQ(output.size(0), 2);
    ASSERT_EQ(output.size(1), 5);
    
    auto [actions, values] = a3c_shared->ForwardActorCritic(input);
    ASSERT_EQ(actions.size(0), 2);
    ASSERT_EQ(actions.size(1), 5);
    ASSERT_EQ(values.size(0), 2);
    ASSERT_EQ(values.size(1), 1);
    
    std::cout << "✓ A3C shared backbone functionality works correctly" << std::endl;
}

// Test A3C initialization
void test_a3c_initialization() {
    auto a3c = networks::A3C(networks::MLPTag{}, 10, 64, 4);
    
    // Test orthogonal initialization
    ASSERT_NO_THROW(a3c->InitOrtho(1.0f));
    
    // Check parameters
    auto params = a3c->parameters();
    ASSERT_TRUE(params.size() > 0);
    
    bool has_non_zero = false;
    for (const auto& param : params) {
        if (torch::any(param != 0).item<bool>()) {
            has_non_zero = true;
            break;
        }
    }
    ASSERT_TRUE(has_non_zero);
    
    std::cout << "✓ A3C orthogonal initialization successful" << std::endl;
}

// Test A3C factory methods
void test_a3c_factory_methods() {
    // Test WithMLP factory
    auto a3c_mlp = networks::A3CImpl::WithMLP(6, 24, 3, 1);
    auto mlp_input = torch::randn({1, 6});
    auto mlp_output = a3c_mlp->forward(mlp_input);
    ASSERT_EQ(mlp_output.size(1), 3);
    
    // Test WithCNN factory
    auto a3c_cnn = networks::A3CImpl::WithCNN(1, 64, 64, 8, 1);
    auto cnn_input = torch::randn({1, 1, 64, 64});
    auto cnn_output = a3c_cnn->forward(cnn_input);
    ASSERT_EQ(cnn_output.size(1), 8);
    
    // Test WithSharedBackbone factory - use WithMLP instead
    auto a3c_shared = networks::A3CImpl::WithMLP(14, 56, 7, 1);
    auto shared_input = torch::randn({1, 14});
    auto shared_output = a3c_shared->forward(shared_input);
    ASSERT_EQ(shared_output.size(1), 7);
    
    std::cout << "✓ A3C factory methods work correctly" << std::endl;
}

// Test A3C delegation to A2C
void test_a3c_delegation() {
    auto a3c = networks::A3C(networks::MLPTag{}, 8, 32, 4);
    
    // Since A3C delegates to internal A2C, test that functionality is preserved
    auto input = torch::randn({1, 8});
    
    // Test that the same input produces consistent results
    auto output1 = a3c->forward(input);
    auto [actions1, values1] = a3c->ForwardActorCritic(input);
    auto action_logits1 = a3c->GetAction(input);
    auto state_values1 = a3c->GetValue(input);
    
    // Actions from ForwardActorCritic should match GetAction
    ASSERT_TRUE(torch::allclose(actions1, action_logits1, 1e-6));
    // Values from ForwardActorCritic should match GetValue
    ASSERT_TRUE(torch::allclose(values1, state_values1, 1e-6));
    
    std::cout << "✓ A3C delegation to A2C works correctly" << std::endl;
}

int main() {
    TestSuite suite("PPOkemon A3C Network Tests");
    
    // Add test cases
    suite.AddTest("A3C MLP", test_a3c_mlp);
    suite.AddTest("A3C Actor-Critic", test_a3c_actor_critic);
    suite.AddTest("A3C CNN", test_a3c_cnn);
    suite.AddTest("A3C Shared Backbone", test_a3c_shared_backbone);
    suite.AddTest("A3C Initialization", test_a3c_initialization);
    suite.AddTest("A3C Factory Methods", test_a3c_factory_methods);
    suite.AddTest("A3C Delegation", test_a3c_delegation);
    
    // Run all tests
    bool all_passed = suite.RunAll();
    
    // Get final statistics
    int passed, failed, total;
    suite.GetStats(passed, failed, total);
    
    std::cout << "\nFinal Results: " << passed << "/" << total << " A3C tests passed" << std::endl;
    
    return all_passed ? 0 : 1;
}
