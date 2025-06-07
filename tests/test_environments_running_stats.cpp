#include <iostream>
#include <torch/torch.h>
#include "torch/utils/running_mean_std.h"
#include "test_utils.h"

using namespace PPOkemonTest;

void test_running_mean_std_basic() {
    RunningMeanStd rms({2}, 1e-8f);
    
    // Create test data with known statistics
    auto data1 = torch::tensor({{1.0f, 2.0f}});
    auto data2 = torch::tensor({{3.0f, 4.0f}});
    auto data3 = torch::tensor({{5.0f, 6.0f}});
    
    rms.Update(data1);
    rms.Update(data2);
    rms.Update(data3);
    
    auto mean = rms.GetMean();
    auto var = rms.GetVar();
    
    // Verify expected values (mean should be [3, 4], variance should be [2.6667, 2.6667])
    auto expected_mean = torch::tensor({3.0f, 4.0f});
    auto expected_var = torch::tensor({2.6667f, 2.6667f});
    
    ASSERT_TRUE(torch::allclose(mean, expected_mean, 1e-3));
    ASSERT_TRUE(torch::allclose(var, expected_var, 1e-3));
}

void test_running_mean_std_batch_update() {
    RunningMeanStd rms({2}, 1e-8f);
    RunningMeanStd rms_batch({2}, 1e-8f);
    
    // Individual updates
    auto data1 = torch::tensor({{1.0f, 2.0f}});
    auto data2 = torch::tensor({{3.0f, 4.0f}});
    auto data3 = torch::tensor({{5.0f, 6.0f}});
    
    rms.Update(data1);
    rms.Update(data2);
    rms.Update(data3);
    
    // Batch update
    auto batch_data = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
    rms_batch.Update(batch_data);
    
    // Verify batch and individual updates produce same results
    ASSERT_TRUE(torch::allclose(rms.GetMean(), rms_batch.GetMean(), 1e-6));
    ASSERT_TRUE(torch::allclose(rms.GetVar(), rms_batch.GetVar(), 1e-6));
}

void test_running_mean_std_save_load() {
    RunningMeanStd rms({2}, 1e-8f);
    
    // Add some data
    auto data = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
    rms.Update(data);
    
    // Save/Load functionality
    std::string test_path = "/tmp/test_rms_stats.bin";
    rms.Save(test_path);
    
    RunningMeanStd rms_loaded({2}, 1e-8f);
    rms_loaded.Load(test_path);
    
    ASSERT_TRUE(torch::allclose(rms.GetMean(), rms_loaded.GetMean(), 1e-6));
    ASSERT_TRUE(torch::allclose(rms.GetVar(), rms_loaded.GetVar(), 1e-6));
    
    // Cleanup
    std::remove(test_path.c_str());
}

void test_running_mean_std_normalization() {
    RunningMeanStd rms({3}, 1e-8f);
    
    // Create deterministic data with known mean and variance
    torch::manual_seed(42);  // Set seed for reproducibility
    auto data = torch::randn({1000, 3}) * 2.0f + 5.0f; // mean≈5, std≈2
    rms.Update(data);
    
    // Manual normalization since Normalize method doesn't exist in RunningMeanStd
    auto mean = rms.GetMean();
    auto var = rms.GetVar();
    auto normalized = (data - mean) / torch::sqrt(var + 1e-8f);
    auto normalized_mean = normalized.mean(0);
    auto normalized_std = normalized.std(0);
    
    // Debug output
    // std::cout << "Original data mean: " << data.mean(0) << std::endl;
    // std::cout << "Original data std: " << data.std(0) << std::endl;
    // std::cout << "RMS mean: " << mean << std::endl;
    // std::cout << "RMS var: " << var << std::endl;
    // std::cout << "RMS std: " << torch::sqrt(var) << std::endl;
    // std::cout << "Normalized mean: " << normalized_mean << std::endl;
    // std::cout << "Normalized std: " << normalized_std << std::endl;
    
    // Check that normalized mean is close to zero (absolute tolerance since we're comparing to zero)
    ASSERT_TRUE(torch::allclose(normalized_mean, torch::zeros({3}), 1e-5, 1e-5));
    ASSERT_TRUE(torch::allclose(normalized_std, torch::ones({3}), 1e-3, 1e-3));
}

int main() {
    TestSuite suite("PPOkemon Environment Running Stats Tests");
    
    suite.AddTest("Basic Functionality", test_running_mean_std_basic);
    suite.AddTest("Batch Update", test_running_mean_std_batch_update);
    suite.AddTest("Save and Load", test_running_mean_std_save_load);
    suite.AddTest("Normalization", test_running_mean_std_normalization);
    
    bool all_passed = suite.RunAll();
    
    int passed, failed, total;
    suite.GetStats(passed, failed, total);
    
    std::cout << "\nFinal Results: " << passed << "/" << total << " tests passed" << std::endl;
    
    return all_passed ? 0 : 1;
}
