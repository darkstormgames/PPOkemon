#include "../include/torch/networks/network_base.h"
#include "../include/torch/networks/mlp.h"
#include "../include/torch/networks/cnn.h"
#include "../include/torch/networks/a2c.h"
#include "../include/torch/networks/a3c.h"
#include "test_utils.h"
#include <iostream>

using namespace PPOkemonTest;

// Quick integration test for all network types
void test_network_integration() {
    std::cout << "Testing network integration..." << std::endl;
    
    // Test MLP
    auto mlp = networks::MLP(8, 32, 4);
    auto mlp_input = torch::randn({2, 8});
    auto mlp_output = mlp->forward(mlp_input);
    ASSERT_EQ(mlp_output.size(0), 2);
    ASSERT_EQ(mlp_output.size(1), 4);
    
    // Test CNN
    auto cnn = networks::CNNBody(1, 160, 144);
    auto img_input = torch::randn({1, 1, 160, 144});
    auto cnn_output = cnn->forward(img_input);
    ASSERT_EQ(cnn_output.size(0), 1);
    ASSERT_TRUE(cnn_output.size(1) > 0);
    
    // Test A2C
    auto a2c = networks::A2C(networks::MLPTag{}, 8, 32, 4);
    auto a2c_output = a2c->forward(mlp_input);
    ASSERT_EQ(a2c_output.size(0), 2);
    ASSERT_EQ(a2c_output.size(1), 4);
    
    // Test A3C  
    auto a3c = networks::A3C(networks::MLPTag{}, 8, 32, 4);
    auto a3c_output = a3c->forward(mlp_input);
    ASSERT_EQ(a3c_output.size(0), 2);
    ASSERT_EQ(a3c_output.size(1), 4);
    
    std::cout << "✓ All network types work correctly" << std::endl;
}

// Test cross-network compatibility
void test_network_compatibility() {
    std::cout << "Testing network compatibility..." << std::endl;
    
    // Create shared backbone for testing - use MLPImpl directly
    auto mlp_impl = std::make_shared<networks::MLPImpl>(6, 24, 12);
    std::shared_ptr<networks::NetworkBase> backbone = std::static_pointer_cast<networks::NetworkBase>(mlp_impl);
    auto a2c_shared = networks::A2C(networks::SharedBackboneTag{}, backbone, 3, 1);
    auto a3c_shared = networks::A3C(networks::SharedBackboneTag{}, backbone, 3, 1);
    
    auto input = torch::randn({1, 6});
    
    auto a2c_output = a2c_shared->forward(input);
    auto a3c_output = a3c_shared->forward(input);
    
    ASSERT_EQ(a2c_output.size(1), 3);
    ASSERT_EQ(a3c_output.size(1), 3);
    
    std::cout << "✓ Network compatibility verified" << std::endl;
}

// Test network initialization consistency
void test_network_initialization() {
    std::cout << "Testing network initialization..." << std::endl;
    
    auto mlp = networks::MLP(4, 16, 2);
    auto a2c = networks::A2C(networks::MLPTag{}, 4, 16, 2);
    auto a3c = networks::A3C(networks::MLPTag{}, 4, 16, 2);
    auto cnn = networks::CNNBody(1, 64, 64);
    
    // Test all support orthogonal initialization
    ASSERT_NO_THROW(mlp->InitOrtho(1.0f));
    ASSERT_NO_THROW(a2c->InitOrtho(1.0f));
    ASSERT_NO_THROW(a3c->InitOrtho(1.0f));
    ASSERT_NO_THROW(cnn->InitOrtho(1.0f));
    
    std::cout << "✓ All networks support orthogonal initialization" << std::endl;
}

int main() {
    TestSuite suite("PPOkemon Network Integration Tests");
    
    std::cout << "Running simplified network integration tests..." << std::endl;
    std::cout << "For detailed tests, run:" << std::endl;
    std::cout << "  - test_networks_mlp for MLP tests" << std::endl;
    std::cout << "  - test_networks_cnn for CNN tests" << std::endl;
    std::cout << "  - test_networks_a2c for A2C tests" << std::endl;
    std::cout << "  - test_networks_a3c for A3C tests" << std::endl;
    std::cout << std::endl;
    
    // Add test cases
    suite.AddTest("Network Integration", test_network_integration);
    suite.AddTest("Network Compatibility", test_network_compatibility);
    suite.AddTest("Network Initialization", test_network_initialization);
    
    // Run all tests
    bool all_passed = suite.RunAll();
    
    // Get final statistics
    int passed, failed, total;
    suite.GetStats(passed, failed, total);
    
    std::cout << "\nFinal Results: " << passed << "/" << total << " integration tests passed" << std::endl;
    
    return all_passed ? 0 : 1;
}
