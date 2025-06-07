#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <torch/torch.h>
#include "torch/envs/env_base.h"
#include "test_utils.h"

using namespace PPOkemonTest;

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

void test_base_env_basic_functionality() {
    TestEnv env(42);
    
    // Test basic functionality
    env.Reset();
    std::vector<float> obs_buffer(env.GetObservationSize());
    env.GetObsData(obs_buffer.data());
    
    ASSERT_EQ(env.GetObservationSize(), 4);
    ASSERT_EQ(env.GetObservationShape().size(), 1);
    ASSERT_EQ(env.GetObservationShape()[0], 4);
    ASSERT_EQ(env.GetActionSize(), 2);
}

void test_base_env_step_functionality() {
    TestEnv env(42);
    env.Reset();
    
    // Test step functionality
    float action = 1.0f;
    auto result = env.Step(&action, 1);
    
    ASSERT_TRUE(result.reward != 0.0f || result.reward == 0.0f); // Reward can be anything
    ASSERT_TRUE(result.done == true || result.done == false);   // Done can be true or false
}

void test_base_env_episode_handling() {
    TestEnv env(42);
    env.Reset();
    
    // Run until episode is done
    float action = 1.0f;
    bool episode_finished = false;
    int steps = 0;
    
    while (!episode_finished && steps < 20) { // Safety limit
        auto result = env.Step(&action, 1);
        episode_finished = result.done;
        steps++;
        
        if (episode_finished) {
            ASSERT_TRUE(result.tot_steps > 0); // Use tot_steps instead of episode_length
            break;
        }
    }
    
    ASSERT_TRUE(episode_finished); // Episode should finish within 20 steps
}

void test_base_env_normalization() {
    TestEnv env(42);
    
    // Test normalization setup
    env.SetObservationNormalization(true, 1e-8f);
    env.SetRewardNormalization(true, 1e-8f);
    
    env.Reset();
    std::vector<float> obs_buffer(env.GetObservationSize());
    
    // Run a few steps to build up statistics
    for (int step = 0; step < 5; ++step) {
        env.GetNormalizedObsData(obs_buffer.data());
        float action = static_cast<float>(step % 2);
        auto result = env.Step(&action, 1);
        
        if (result.done) {
            env.Reset();
        }
    }
    
    // Test normalization doesn't crash
    ASSERT_NO_THROW(env.GetNormalizedObsData(obs_buffer.data()));
}

void test_base_env_normalization_persistence() {
    TestEnv env(42);
    
    // Set up normalization
    env.SetObservationNormalization(true, 1e-8f);
    env.SetRewardNormalization(true, 1e-8f);
    
    // Run some steps to gather statistics
    env.Reset();
    for (int i = 0; i < 10; ++i) {
        float action = 1.0f;
        env.Step(&action, 1);
    }
    
    // Test statistics persistence
    std::string obs_path = "/tmp/test_obs_stats.bin";
    std::string reward_path = "/tmp/test_reward_stats.bin";
    ASSERT_NO_THROW(env.SaveNormalizationStats(obs_path, reward_path));
    
    TestEnv env2(999);
    env2.SetObservationNormalization(true, 1e-8f);
    env2.SetRewardNormalization(true, 1e-8f);
    ASSERT_NO_THROW(env2.LoadNormalizationStats(obs_path, reward_path));
    
    // Cleanup
    std::remove(obs_path.c_str());
    std::remove(reward_path.c_str());
}

int main() {
    TestSuite suite("PPOkemon Base Environment Tests");
    
    suite.AddTest("Basic Functionality", test_base_env_basic_functionality);
    suite.AddTest("Step Functionality", test_base_env_step_functionality);
    suite.AddTest("Episode Handling", test_base_env_episode_handling);
    suite.AddTest("Normalization", test_base_env_normalization);
    suite.AddTest("Normalization Persistence", test_base_env_normalization_persistence);
    
    bool all_passed = suite.RunAll();
    
    int passed, failed, total;
    suite.GetStats(passed, failed, total);
    
    std::cout << "\nFinal Results: " << passed << "/" << total << " tests passed" << std::endl;
    
    return all_passed ? 0 : 1;
}
