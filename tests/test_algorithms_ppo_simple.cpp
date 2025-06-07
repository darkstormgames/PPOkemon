#include "../include/torch/algorithms/ppo.h"
#include "../include/torch/networks/a2c.h"
#include "../include/torch/envs/env_abstract.h"
#include "test_utils.h"
#include <iostream>
#include <memory>
#include <filesystem>

using namespace PPOkemonTest;
using namespace algorithms;

// Simple mock environment for PPO testing
class SimpleMockEnv : public AbstractEnv {
private:
    int step_count_;
    int max_steps_;
    std::vector<float> obs_;

public:
    SimpleMockEnv() : AbstractEnv(0), step_count_(0), max_steps_(100) {
        obs_ = {0.5f, -0.3f, 0.1f, 0.8f};
    }

protected:
    void ResetImpl() override {
        step_count_ = 0;
        obs_ = {0.5f, -0.3f, 0.1f, 0.8f};
    }

    StepResultRaw StepImpl(const float* actions, int64_t action_size) override {
        (void)actions;
        (void)action_size;
        
        step_count_++;
        StepResultRaw result;
        result.reward = 1.0f;
        result.done = (step_count_ >= max_steps_);
        result.tot_reward = step_count_ * 1.0f;
        result.tot_steps = step_count_;
        
        return result;
    }

    void RenderImpl() override {
        // No rendering needed
    }

public:
    int64_t GetObservationSize() const override {
        return 4;
    }

    int64_t GetActionSize() const override {
        return 2;
    }

    void GetObsData(float* buffer) const override {
        for (size_t i = 0; i < obs_.size(); ++i) {
            buffer[i] = obs_[i];
        }
    }
};

// Test basic PPO initialization
void test_ppo_basic_initialization() {
    std::shared_ptr<networks::NetworkBase> network = networks::A2CImpl::WithMLP(4, 32, 2);
    
    PPO::Config config;
    config.device = torch::kCPU;
    config.num_envs = 1;
    config.rollout_steps = 8;
    config.mini_batch_size = 4;
    config.ppo_epochs = 2;
    
    // Test that we can create PPO without errors
    ASSERT_NO_THROW({
        auto ppo = std::make_shared<PPO>(network, config);
        ASSERT_TRUE(ppo != nullptr);
    });
    
    std::cout << "PPO basic initialization successful" << std::endl;
}

// Test PPO config values
void test_ppo_config_values() {
    PPO::Config config;
    
    // Test default values
    ASSERT_EQ(config.clip_ratio, 0.2f);
    ASSERT_EQ(config.value_clip_ratio, 0.2f);
    ASSERT_EQ(config.entropy_coef, 0.01f);
    ASSERT_EQ(config.value_coef, 0.5f);
    ASSERT_EQ(config.ppo_epochs, 4);
    ASSERT_EQ(config.mini_batch_size, 64);
    ASSERT_EQ(config.learning_rate, 3e-4f);
    
    std::cout << "PPO config values verified" << std::endl;
}

// Test that PPO can be created with different network sizes
void test_ppo_different_networks() {
    PPO::Config config;
    config.device = torch::kCPU;
    config.num_envs = 1;
    
    // Test with different observation sizes
    std::shared_ptr<networks::NetworkBase> network1 = networks::A2CImpl::WithMLP(8, 64, 4);
    ASSERT_NO_THROW({
        auto ppo1 = std::make_shared<PPO>(network1, config);
    });
    
    std::shared_ptr<networks::NetworkBase> network2 = networks::A2CImpl::WithMLP(16, 128, 6);
    ASSERT_NO_THROW({
        auto ppo2 = std::make_shared<PPO>(network2, config);
    });
    
    std::cout << "PPO works with different network configurations" << std::endl;
}

// Test environment creation
void test_environment_creation() {
    auto env = std::make_unique<SimpleMockEnv>();
    
    ASSERT_EQ(env->GetObservationSize(), 4);
    ASSERT_EQ(env->GetActionSize(), 2);
    
    // Test reset
    env->Reset();
    
    // Test step
    float actions[2] = {0.5f, -0.5f};
    auto result = env->Step(actions, 2);
    
    ASSERT_EQ(result.reward, 1.0f);
    ASSERT_FALSE(result.done); // Should not be done after 1 step
    ASSERT_EQ(result.tot_steps, 1);
    
    std::cout << "Mock environment creation and basic operations successful" << std::endl;
}

int main() {
    TestSuite suite("PPO Algorithm Basic Tests");
    
    // Add basic test cases that should work
    suite.AddTest("PPO Basic Initialization", test_ppo_basic_initialization);
    suite.AddTest("PPO Config Values", test_ppo_config_values);
    suite.AddTest("PPO Different Networks", test_ppo_different_networks);
    suite.AddTest("Environment Creation", test_environment_creation);
    
    // Run all tests
    bool all_passed = suite.RunAll();
    
    // Get final statistics
    int passed, failed, total;
    suite.GetStats(passed, failed, total);
    
    std::cout << "\nFinal Results: " << passed << "/" << total << " tests passed" << std::endl;
    
    return all_passed ? 0 : 1;
}
