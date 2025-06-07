/**
 * @file test_environments.cpp
 * @brief Main environment test suite that runs all environment-related tests
 * 
 * This file serves as a comprehensive test runner for all environment components.
 * For focused testing, use the individual test files:
 * - test_environments_base.cpp: Base environment functionality
 * - test_environments_vectorized.cpp: Vectorized environment tests
 * - test_environments_running_stats.cpp: Running statistics tests
 */

#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <torch/torch.h>
#include "torch/utils/running_mean_std.h"
#include "torch/envs/env_base.h"
#include "torch/envs/env_vectorized.h"
#include "test_utils.h"

using namespace PPOkemonTest;

// Simple test environment for integrated testing
class IntegratedTestEnv : public BaseEnv
{
private:
    std::vector<int64_t> obs_shape_;
    int64_t obs_size_;
    mutable std::normal_distribution<float> noise_dist_;

public:
    IntegratedTestEnv(unsigned int seed = 0) 
        : BaseEnv(seed)
        , obs_shape_({4})
        , obs_size_(4)
        , noise_dist_(0.0f, 1.0f)
    {
    }

    std::vector<int64_t> GetObservationShape() const override {
        return obs_shape_;
    }

    int64_t GetObservationSize() const override {
        return obs_size_;
    }

    int64_t GetActionSize() const override {
        return 2;
    }

    void GetObsData(float* buffer) const override {
        auto& rng = const_cast<std::mt19937&>(random_engine);
        for (int i = 0; i < obs_size_; ++i) {
            buffer[i] = const_cast<std::normal_distribution<float>&>(noise_dist_)(rng);
        }
    }

protected:
    StepResultRaw StepImpl(const float* action_data, int64_t action_size) override {
        (void)action_data; (void)action_size;
        current_episode_length++;
        float reward = noise_dist_(random_engine) * 0.1f;
        current_episode_reward += reward;
        
        bool done = (current_episode_length >= 10);
        
        return {reward, done, current_episode_reward, current_episode_length};
    }

    void ResetImpl() override {
        // Base class handles episode counters
    }

    void RenderImpl() override {
        // Silent for testing
    }
};

// Quick integration test for running statistics
void test_running_mean_std_integration() {
    RunningMeanStd rms({2}, 1e-8f);
    
    auto data = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
    rms.Update(data);
    
    auto mean = rms.GetMean();
    auto var = rms.GetVar();
    
    ASSERT_TRUE(torch::allclose(mean, torch::tensor({3.0f, 4.0f}), 1e-3));
    ASSERT_TRUE(var.sum().item<float>() > 0); // Variance should be positive
}

// Quick integration test for base environment
void test_base_env_integration() {
    IntegratedTestEnv env(42);
    
    env.Reset();
    std::vector<float> obs_buffer(env.GetObservationSize());
    env.GetObsData(obs_buffer.data());
    
    ASSERT_EQ(env.GetObservationSize(), 4);
    ASSERT_EQ(env.GetActionSize(), 2);
    
    float action = 1.0f;
    auto result = env.Step(&action, 1);
    ASSERT_TRUE(result.tot_steps > 0);  // Use tot_steps instead of episode_length
}

// Quick integration test for vectorized environment
void test_vectorized_env_integration() {
    const int num_envs = 2;
    
    auto env_factory = []() -> std::unique_ptr<AbstractEnv> {
        return std::make_unique<IntegratedTestEnv>(std::random_device{}());
    };
    
    VectorizedEnv vec_env(env_factory, num_envs, torch::kCPU);
    
    auto obs_batch = vec_env.Reset();
    ASSERT_EQ(obs_batch.size(0), num_envs);
    ASSERT_EQ(obs_batch.size(1), 4);
    
    auto actions = torch::ones({num_envs, 2}, torch::kFloat32);
    auto [next_obs, step_results] = vec_env.Step(actions);
    
    ASSERT_EQ(step_results.rewards.size(0), num_envs);
    ASSERT_EQ(step_results.dones.size(0), num_envs);
}

int main() {
    TestSuite suite("PPOkemon Environment Integration Tests");
    
    std::cout << "=== Running Environment Integration Tests ===" << std::endl;
    std::cout << "For detailed tests, run:" << std::endl;
    std::cout << "  - test_environments_base: Base environment functionality" << std::endl;
    std::cout << "  - test_environments_vectorized: Vectorized environment tests" << std::endl;
    std::cout << "  - test_environments_running_stats: Running statistics tests" << std::endl;
    std::cout << std::endl;
    
    // Add integration test cases
    suite.AddTest("RunningMeanStd Integration", test_running_mean_std_integration);
    suite.AddTest("BaseEnv Integration", test_base_env_integration);
    suite.AddTest("VectorizedEnv Integration", test_vectorized_env_integration);
    
    // Run all tests
    bool all_passed = suite.RunAll();
    
    // Get final statistics
    int passed, failed, total;
    suite.GetStats(passed, failed, total);
    
    std::cout << "\nFinal Results: " << passed << "/" << total << " integration tests passed" << std::endl;
    
    return all_passed ? 0 : 1;
}
