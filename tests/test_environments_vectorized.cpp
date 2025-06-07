#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <torch/torch.h>
#include "torch/envs/env_base.h"
#include "torch/envs/env_vectorized.h"
#include "test_utils.h"

using namespace PPOkemonTest;

// Simple test environment for vectorized testing
class SimpleTestEnv : public AbstractEnv
{
private:
    std::vector<int64_t> obs_shape_;
    int64_t obs_size_;
    mutable std::normal_distribution<float> noise_dist_;

public:
    SimpleTestEnv(unsigned int seed = 0) 
        : AbstractEnv(seed)
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
        
        bool done = (current_episode_length >= 10); // Short episodes
        
        return {reward, done, current_episode_reward, current_episode_length};
    }

    void ResetImpl() override {
        // Base class handles episode counters
    }

    void RenderImpl() override {
        // Silent for testing
    }
};

void test_vectorized_env_creation() {
    const int num_envs = 4;
    
    // Create environment factory function
    auto env_factory = []() -> std::unique_ptr<AbstractEnv> {
        return std::make_unique<SimpleTestEnv>(std::random_device{}());
    };
    
    VectorizedEnv vec_env(env_factory, num_envs, torch::kCPU);
    
    // Test that vectorized environment was created successfully
    ASSERT_NO_THROW(vec_env.Reset());
}

void test_vectorized_env_reset() {
    const int num_envs = 4;
    
    auto env_factory = []() -> std::unique_ptr<AbstractEnv> {
        return std::make_unique<SimpleTestEnv>(std::random_device{}());
    };
    
    VectorizedEnv vec_env(env_factory, num_envs, torch::kCPU);
    
    // Test reset
    auto obs_batch = vec_env.Reset();
    ASSERT_EQ(obs_batch.size(0), num_envs);
    ASSERT_EQ(obs_batch.size(1), 4); // observation size
}

void test_vectorized_env_step() {
    const int num_envs = 4;
    
    auto env_factory = []() -> std::unique_ptr<AbstractEnv> {
        return std::make_unique<SimpleTestEnv>(std::random_device{}());
    };
    
    VectorizedEnv vec_env(env_factory, num_envs, torch::kCPU);
    
    // Reset environments
    auto obs_batch = vec_env.Reset();
    
    // Test step with proper tensor actions (action_size = 2)
    auto actions = torch::randint(0, 2, {num_envs, 2}, torch::kFloat32);
    auto [next_obs, step_results] = vec_env.Step(actions);
    
    ASSERT_EQ(step_results.rewards.size(0), num_envs);
    ASSERT_EQ(step_results.dones.size(0), num_envs);
    ASSERT_EQ(next_obs.size(0), num_envs);
    ASSERT_EQ(next_obs.size(1), 4);
}

void test_vectorized_env_multiple_steps() {
    const int num_envs = 3;
    
    auto env_factory = []() -> std::unique_ptr<AbstractEnv> {
        return std::make_unique<SimpleTestEnv>(std::random_device{}());
    };
    
    VectorizedEnv vec_env(env_factory, num_envs, torch::kCPU);
    
    // Reset environments
    vec_env.Reset();
    
    // Test multiple steps
    for (int i = 0; i < 5; ++i) {
        auto random_actions = torch::randint(0, 2, {num_envs, 2}, torch::kFloat32);
        auto [obs, results] = vec_env.Step(random_actions);
        
        ASSERT_EQ(results.rewards.size(0), num_envs);
        ASSERT_EQ(results.dones.size(0), num_envs);
        ASSERT_EQ(obs.size(0), num_envs);
        ASSERT_EQ(obs.size(1), 4);
    }
}

void test_vectorized_env_episode_completion() {
    const int num_envs = 2;
    
    auto env_factory = []() -> std::unique_ptr<AbstractEnv> {
        return std::make_unique<SimpleTestEnv>(42); // Fixed seed for deterministic test
    };
    
    VectorizedEnv vec_env(env_factory, num_envs, torch::kCPU);
    vec_env.Reset();
    
    // Run until at least one episode is done
    bool any_done = false;
    int steps = 0;
    const int max_steps = 20;
    
    while (!any_done && steps < max_steps) {
        auto actions = torch::ones({num_envs, 2}, torch::kFloat32);
        auto [obs, results] = vec_env.Step(actions);
        
        // Check if any environment is done
        for (int i = 0; i < num_envs; ++i) {
            if (results.dones[i].item<bool>()) {
                any_done = true;
                break;
            }
        }
        steps++;
    }
    
    // At least one episode should finish within max_steps
    ASSERT_TRUE(any_done);
}

int main() {
    TestSuite suite("PPOkemon Vectorized Environment Tests");
    
    suite.AddTest("Creation", test_vectorized_env_creation);
    suite.AddTest("Reset", test_vectorized_env_reset);
    suite.AddTest("Step", test_vectorized_env_step);
    suite.AddTest("Multiple Steps", test_vectorized_env_multiple_steps);
    suite.AddTest("Episode Completion", test_vectorized_env_episode_completion);
    
    bool all_passed = suite.RunAll();
    
    int passed, failed, total;
    suite.GetStats(passed, failed, total);
    
    std::cout << "\nFinal Results: " << passed << "/" << total << " tests passed" << std::endl;
    
    return all_passed ? 0 : 1;
}
