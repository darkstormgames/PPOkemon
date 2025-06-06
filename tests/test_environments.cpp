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

// Test RunningMeanStd implementation
void test_running_mean_std() {
    // Test 1: Basic functionality
    RunningMeanStd rms({2}, 1e-8f);
    
    // Create some test data with known statistics
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
    
    // Test 2: Batch update
    auto batch_data = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
    RunningMeanStd rms_batch({2}, 1e-8f);
    rms_batch.Update(batch_data);
    
    // Verify batch and individual updates produce same results
    ASSERT_TRUE(torch::allclose(rms.GetMean(), rms_batch.GetMean(), 1e-6));
    ASSERT_TRUE(torch::allclose(rms.GetVar(), rms_batch.GetVar(), 1e-6));
    
    // Test 3: Save/Load functionality
    std::string test_path = "/tmp/test_rms_stats.bin";
    rms.Save(test_path);
    
    RunningMeanStd rms_loaded({2}, 1e-8f);
    rms_loaded.Load(test_path);
    
    ASSERT_TRUE(torch::allclose(rms.GetMean(), rms_loaded.GetMean(), 1e-6));
    ASSERT_TRUE(torch::allclose(rms.GetVar(), rms_loaded.GetVar(), 1e-6));
}

// Example concrete environment for testing BaseEnv
class TestEnv : public BaseEnv
{
private:
    std::vector<int64_t> obs_shape_;
    int64_t obs_size_;
    mutable std::normal_distribution<float> noise_dist_;

public:
    TestEnv(unsigned int seed = 0) 
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
        return 2;  // Two actions for testing
    }

    void GetObsData(float* buffer) const override {
        // Create a local copy of the random engine for const methods
        auto& rng = const_cast<std::mt19937&>(random_engine);
        for (int i = 0; i < obs_size_; ++i) {
            buffer[i] = const_cast<std::normal_distribution<float>&>(noise_dist_)(rng);
        }
    }

protected:
    StepResultRaw StepImpl(const float* action_data, int64_t action_size) override {
        (void)action_data; // Suppress unused parameter warning
        (void)action_size; // Suppress unused parameter warning
        current_episode_length++;
        float reward = noise_dist_(random_engine) * 0.1f; // Small rewards for testing
        current_episode_reward += reward;
        
        bool done = (current_episode_length >= 10); // Short episodes for testing
        
        return {reward, done, current_episode_reward, current_episode_length};
    }

    void ResetImpl() override {
        // AbstractEnv::Reset() handles the episode counters
    }

    void RenderImpl() override {
        // Silent for testing
    }
};

// Test BaseEnv functionality
void test_base_env() {
    TestEnv env(42);
    
    // Test basic functionality
    env.Reset();
    std::vector<float> obs_buffer(env.GetObservationSize());
    env.GetObsData(obs_buffer.data());
    
    // Test normalization setup
    env.SetObservationNormalization(true, 1e-8f);
    env.SetRewardNormalization(true, 1e-8f);
    
    // Run a few steps to build up statistics
    for (int step = 0; step < 5; ++step) {
        env.GetNormalizedObsData(obs_buffer.data());
        float action = static_cast<float>(step % 2);
        auto result = env.Step(&action, 1);
        
        if (result.done) {
            env.Reset();
        }
    }
    
    // Test statistics persistence
    std::string obs_path = "/tmp/test_obs_stats.bin";
    std::string reward_path = "/tmp/test_reward_stats.bin";
    env.SaveNormalizationStats(obs_path, reward_path);
    
    TestEnv env2(999);
    env2.SetObservationNormalization(true, 1e-8f);
    env2.SetRewardNormalization(true, 1e-8f);
    env2.LoadNormalizationStats(obs_path, reward_path);
    
    // Basic assertions to ensure environment works
    ASSERT_EQ(env.GetObservationSize(), 4);
    ASSERT_EQ(env.GetObservationShape().size(), 1);
    ASSERT_EQ(env.GetObservationShape()[0], 4);
}

// Test VectorizedEnv functionality
void test_vectorized_env() {
    const int num_envs = 4;
    
    // Create environment factory function
    auto env_factory = []() -> std::unique_ptr<AbstractEnv> {
        return std::make_unique<TestEnv>(std::random_device{}());
    };
    
    VectorizedEnv vec_env(env_factory, num_envs, torch::kCPU);
    
    // Test reset
    auto obs_batch = vec_env.Reset();
    ASSERT_EQ(obs_batch.size(0), num_envs);
    ASSERT_EQ(obs_batch.size(1), 4);
    
    // Test step with proper tensor actions (action_size = 2 for TestEnv)
    auto actions = torch::randint(0, 2, {num_envs, 2}, torch::kFloat32);
    auto [next_obs, step_results] = vec_env.Step(actions);
    
    ASSERT_EQ(step_results.rewards.size(0), num_envs);
    ASSERT_EQ(step_results.dones.size(0), num_envs);
    ASSERT_EQ(next_obs.size(0), num_envs);
    ASSERT_EQ(next_obs.size(1), 4);
    
    // Test multiple steps
    for (int i = 0; i < 3; ++i) {
        auto random_actions = torch::randint(0, 2, {num_envs, 2}, torch::kFloat32);
        auto [obs, results] = vec_env.Step(random_actions);
        ASSERT_EQ(results.rewards.size(0), num_envs);
        ASSERT_EQ(results.dones.size(0), num_envs);
        ASSERT_EQ(obs.size(0), num_envs);
        ASSERT_EQ(obs.size(1), 4);
    }
}

int main() {
    TestSuite suite("PPOkemon Environment Tests");
    
    // Add test cases
    suite.AddTest("RunningMeanStd", test_running_mean_std);
    suite.AddTest("BaseEnv", test_base_env);
    suite.AddTest("VectorizedEnv", test_vectorized_env);
    
    // Run all tests
    bool all_passed = suite.RunAll();
    
    // Get final statistics
    int passed, failed, total;
    suite.GetStats(passed, failed, total);
    
    std::cout << "\nFinal Results: " << passed << "/" << total << " tests passed" << std::endl;
    
    return all_passed ? 0 : 1;
}
