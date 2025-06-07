#include "../include/torch/networks/network_base.h"
#include "../include/torch/networks/cnn.h"
#include "test_utils.h"
#include <iostream>

using namespace PPOkemonTest;

// Test CNN Network basic functionality
void test_cnn_basic() {
    auto cnn = networks::CNNBody(1, 160, 144);  // Gameboy screen dimensions
    auto img_input = torch::randn({1, 1, 160, 144});
    auto cnn_output = cnn->forward(img_input);
    
    ASSERT_EQ(cnn_output.size(0), 1);
    ASSERT_TRUE(cnn_output.size(1) > 0);  // Output should have some features
    
    std::cout << "✓ CNN forward pass: input [1,1,160,144] -> output [" << cnn_output.size(0) << "," << cnn_output.size(1) << "]" << std::endl;
}

// Test CNN with batch input
void test_cnn_batch() {
    auto cnn = networks::CNNBody(3, 84, 84);  // RGB input, Atari-like dimensions
    auto batch_input = torch::randn({4, 3, 84, 84});  // Batch of 4 RGB images
    auto batch_output = cnn->forward(batch_input);
    
    ASSERT_EQ(batch_output.size(0), 4);
    ASSERT_TRUE(batch_output.size(1) > 0);
    
    std::cout << "✓ CNN batch processing: input [4,3,84,84] -> output [" << batch_output.size(0) << "," << batch_output.size(1) << "]" << std::endl;
}

// Test CNN orthogonal initialization
void test_cnn_initialization() {
    auto cnn = networks::CNNBody(1, 160, 144);
    
    // Test orthogonal initialization
    ASSERT_NO_THROW(cnn->InitOrtho(1.0f));
    
    // Check that parameters exist and are initialized
    auto params = cnn->parameters();
    ASSERT_TRUE(params.size() > 0);
    
    bool has_non_zero = false;
    for (const auto& param : params) {
        if (torch::any(param != 0).item<bool>()) {
            has_non_zero = true;
            break;
        }
    }
    ASSERT_TRUE(has_non_zero);
    
    std::cout << "✓ CNN orthogonal initialization successful" << std::endl;
}

// Test CNN output size method
void test_cnn_output_size() {
    auto cnn = networks::CNNBody(1, 160, 144);
    int64_t output_size = cnn->GetOutputSize();
    
    ASSERT_TRUE(output_size > 0);
    
    // Verify output size matches actual forward pass
    auto test_input = torch::randn({1, 1, 160, 144});
    auto test_output = cnn->forward(test_input);
    ASSERT_EQ(output_size, test_output.size(1));
    
    std::cout << "✓ CNN output size: " << output_size << " features" << std::endl;
}

// Test CNN with different input sizes
void test_cnn_various_inputs() {
    // Test grayscale
    auto cnn_gray = networks::CNNBody(1, 64, 64);
    auto gray_input = torch::randn({2, 1, 64, 64});
    auto gray_output = cnn_gray->forward(gray_input);
    ASSERT_EQ(gray_output.size(0), 2);
    
    // Test RGB
    auto cnn_rgb = networks::CNNBody(3, 128, 128);
    auto rgb_input = torch::randn({2, 3, 128, 128});
    auto rgb_output = cnn_rgb->forward(rgb_input);
    ASSERT_EQ(rgb_output.size(0), 2);
    
    std::cout << "✓ CNN various input formats work correctly" << std::endl;
}

int main() {
    TestSuite suite("PPOkemon CNN Network Tests");
    
    // Add test cases
    suite.AddTest("CNN Basic Functionality", test_cnn_basic);
    suite.AddTest("CNN Batch Processing", test_cnn_batch);
    suite.AddTest("CNN Initialization", test_cnn_initialization);
    suite.AddTest("CNN Output Size", test_cnn_output_size);
    suite.AddTest("CNN Various Inputs", test_cnn_various_inputs);
    
    // Run all tests
    bool all_passed = suite.RunAll();
    
    // Get final statistics
    int passed, failed, total;
    suite.GetStats(passed, failed, total);
    
    std::cout << "\nFinal Results: " << passed << "/" << total << " CNN tests passed" << std::endl;
    
    return all_passed ? 0 : 1;
}
