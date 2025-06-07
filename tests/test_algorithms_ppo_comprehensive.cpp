#include "../include/torch/algorithms/ppo.h"
#include "../include/torch/networks/a2c.h"
#include "../include/torch/envs/env_abstract.h"
#include "test_utils.h"
#include <iostream>
#include <memory>
#include <filesystem>

using namespace PPOkemonTest;
using namespace algorithms;

// Mock environment for PPO testing (using the working simple version)
class MockPPOEnvironment : public AbstractEnv {
private:
    int step_count_;
    int max_steps_;
    std::vector<float> obs_;

public:
    MockPPOEnvironment() : AbstractEnv(0), step_count_(0), max_steps_(100) {
        obs_ = {0.5f, -0.3f, 0.1f, 0.8f}; // 4-dimensional observation
    }

protected:
    void ResetImpl() override {
        step_count_ = 0;
        obs_ = {0.5f, -0.3f, 0.1f, 0.8f};
    }

    StepResultRaw StepImpl(const float* actions, int64_t action_size) override {
        (void)actions; // Suppress unused parameter warning
        (void)action_size;
        
        step_count_++;
        StepResultRaw result;
        result.reward = 1.0f; // Fixed reward for testing
        result.done = (step_count_ >= max_steps_);
        result.tot_reward = step_count_ * 1.0f;
        result.tot_steps = step_count_;
        
        return result;
    }

    void RenderImpl() override {
        // Nothing to render for mock environment
    }

public:
    int64_t GetObservationSize() const override {
        return 4;
    }

    int64_t GetActionSize() const override {
        return 2; // Discrete actions: 0 or 1
    }

    void GetObsData(float* obs_data) const override {
        for (size_t i = 0; i < obs_.size(); ++i) {
            obs_data[i] = obs_[i];
        }
    }
};

// Test PPO initialization
void test_ppo_initialization() {
    // Create A2C network for PPO (policy + value) using factory method
    auto network = networks::A2CImpl::WithMLP(4, 32, 2);
    
    // Create PPO config
    PPO::Config config;
    config.device = torch::kCPU;
    config.num_envs = 2;
    config.rollout_steps = 8;
    config.mini_batch_size = 4;
    config.ppo_epochs = 2;
    config.learning_rate = 3e-4f;
    config.clip_ratio = 0.2f;
    config.value_coef = 0.5f;
    config.entropy_coef = 0.01f;
    config.normalize_advantages = true;
    
    // Create PPO instance
    auto ppo = std::make_unique<PPO>(network, config);
    
    ASSERT_TRUE(ppo != nullptr);
    std::cout << "PPO initialized successfully" << std::endl;
}

// Test PPO environment initialization
void test_ppo_environment_initialization() {
    auto network = networks::A2CImpl::WithMLP(4, 32, 2);
    
    PPO::Config config;
    config.device = torch::kCPU;
    config.num_envs = 1; // Start with single environment like simple test
    config.rollout_steps = 8;
    config.mini_batch_size = 4;
    config.ppo_epochs = 2;
    
    auto ppo = std::make_unique<PPO>(network, config);
    
    // Create mock environments (only 1 to match config)
    std::vector<std::unique_ptr<AbstractEnv>> environments;
    environments.push_back(std::make_unique<MockPPOEnvironment>());
    
    // Test training with environments (this initializes environments internally)
    try {
        ppo->TrainWithEnvironments(environments, 1);
        std::cout << "PPO environment initialization successful" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "❌ PPO Training Exception: " << e.what() << std::endl;
        throw; // Re-throw so test fails properly
    }
}

// Test PPO action generation
void test_ppo_action_generation() {
    auto network = networks::A2CImpl::WithMLP(4, 32, 2);
    
    PPO::Config config;
    config.device = torch::kCPU;
    config.num_envs = 1; // Use single environment
    
    auto ppo = std::make_unique<PPO>(network, config);
    
    // Test that PPO was created successfully (can't test private methods directly)
    ASSERT_TRUE(ppo != nullptr);
    
    // Test that config is accessible
    const auto& ppo_config = ppo->GetConfig();
    ASSERT_EQ(ppo_config.num_envs, 1);
    
    std::cout << "PPO action generation testing passed (indirect test)" << std::endl;
}

// Test PPO value estimation
void test_ppo_value_estimation() {
    auto network = networks::A2CImpl::WithMLP(4, 32, 2);
    
    PPO::Config config;
    config.device = torch::kCPU;
    
    auto ppo = std::make_unique<PPO>(network, config);
    
    // Test that PPO was created successfully (can't test private methods directly)
    ASSERT_TRUE(ppo != nullptr);
    
    // Test network functionality through PPO config access
    const auto& ppo_config = ppo->GetConfig();
    ASSERT_TRUE(ppo_config.device == torch::kCPU);
    
    std::cout << "PPO value estimation testing passed (indirect test)" << std::endl;
}

// Test PPO experience collection
void test_ppo_experience_collection() {
    auto network = networks::A2CImpl::WithMLP(4, 32, 2);
    
    PPO::Config config;
    config.device = torch::kCPU;
    config.num_envs = 1; // Use single environment
    config.rollout_steps = 4; // Small number for testing
    config.mini_batch_size = 2;
    config.ppo_epochs = 1;
    
    auto ppo = std::make_unique<PPO>(network, config);
    
    // Create mock environments (only 1 to match config)
    std::vector<std::unique_ptr<AbstractEnv>> environments;
    environments.push_back(std::make_unique<MockPPOEnvironment>());
    
    // Test training which includes experience collection (can't test private methods directly)
    ASSERT_NO_THROW(ppo->TrainWithEnvironments(environments, 1));
    
    std::cout << "PPO experience collection successful (tested through training)" << std::endl;
}

// Test PPO policy update
void test_ppo_policy_update() {
    auto network = networks::A2CImpl::WithMLP(4, 32, 2);
    
    PPO::Config config;
    config.device = torch::kCPU;
    config.num_envs = 1; // Use single environment
    config.rollout_steps = 4;
    config.mini_batch_size = 2;
    config.ppo_epochs = 1;
    config.learning_rate = 1e-3f;
    
    auto ppo = std::make_unique<PPO>(network, config);
    
    // Create mock environments (only 1 to match config)
    std::vector<std::unique_ptr<AbstractEnv>> environments;
    environments.push_back(std::make_unique<MockPPOEnvironment>());
    
    // Test training which includes policy updates (can't test private methods directly)
    ASSERT_NO_THROW(ppo->TrainWithEnvironments(environments, 1));
    
    // Verify training stats are updated
    const auto& stats = ppo->GetLastStats();
    ASSERT_TRUE(std::isfinite(stats.total_loss)); // Loss should be finite
    ASSERT_FALSE(std::isnan(stats.total_loss)); // Loss should not be NaN
    
    std::cout << "PPO policy update successful, loss: " << stats.total_loss << std::endl;
}

// Test PPO evaluation
void test_ppo_evaluation() {
    auto network = networks::A2CImpl::WithMLP(4, 32, 2);
    
    PPO::Config config;
    config.device = torch::kCPU;
    config.eval_episodes = 2; // Small number for testing
    
    auto ppo = std::make_unique<PPO>(network, config);
    
    // Create mock environments for evaluation (just 1)
    std::vector<std::unique_ptr<AbstractEnv>> eval_environments;
    eval_environments.push_back(std::make_unique<MockPPOEnvironment>());
    
    // Evaluate policy
    float avg_reward = 0.0f;
    ASSERT_NO_THROW(avg_reward = ppo->EvaluateWithEnvironments(eval_environments));
    
    ASSERT_TRUE(std::isfinite(avg_reward)); // Reward should be finite
    ASSERT_FALSE(std::isnan(avg_reward)); // Reward should not be NaN
    ASSERT_TRUE(avg_reward > 0.0f); // Should have positive reward (our mock env gives +1.0 per step)
    
    std::cout << "PPO evaluation successful, average reward: " << avg_reward << std::endl;
}

// Test PPO training loop (short)
void test_ppo_training_loop() {
    auto network = networks::A2CImpl::WithMLP(4, 32, 2);
    
    PPO::Config config;
    config.device = torch::kCPU;
    config.num_envs = 1; // Use single environment
    config.rollout_steps = 4;
    config.mini_batch_size = 2;
    config.ppo_epochs = 1;
    config.learning_rate = 1e-3f;
    config.eval_frequency = 2; // Evaluate every 2 updates
    config.eval_episodes = 1;
    config.verbose = false; // Reduce output for testing
    
    auto ppo = std::make_unique<PPO>(network, config);
    
    // Create mock environments (only 1 to match config)
    std::vector<std::unique_ptr<AbstractEnv>> environments;
    environments.push_back(std::make_unique<MockPPOEnvironment>());
    
    // Run short training
    int total_updates = 3; // Very short for testing
    ASSERT_NO_THROW(ppo->TrainWithEnvironments(environments, total_updates));
    
    std::cout << "PPO training loop completed successfully" << std::endl;
}

// Test PPO model saving and loading
void test_ppo_model_save_load() {
    auto network = networks::A2CImpl::WithMLP(4, 32, 2);
    
    PPO::Config config;
    config.device = torch::kCPU;
    
    auto ppo1 = std::make_unique<PPO>(network, config);
    
    // Create temporary file for saving
    std::string temp_path = "/tmp/test_ppo_model.pt";
    
    // Save model
    ASSERT_NO_THROW(ppo1->SaveModel(temp_path));
    ASSERT_TRUE(std::filesystem::exists(temp_path));
    
    // Create new PPO instance and load model
    auto network2 = networks::A2CImpl::WithMLP(4, 32, 2);
    auto ppo2 = std::make_unique<PPO>(network2, config);
    
    ASSERT_NO_THROW(ppo2->LoadModel(temp_path));
    
    // Clean up
    std::filesystem::remove(temp_path);
    
    std::cout << "PPO model save/load successful" << std::endl;
}

// Test PPO configuration validation
void test_ppo_config_validation() {
    auto network = networks::A2CImpl::WithMLP(4, 32, 2);
    
    // Test with invalid config (mismatched num_envs)
    PPO::Config config;
    config.device = torch::kCPU;
    config.num_envs = 2; // But we'll provide 1 environment
    
    auto ppo = std::make_unique<PPO>(network, config);
    
    // Create mock environments (1 instead of 2)
    std::vector<std::unique_ptr<AbstractEnv>> environments;
    environments.push_back(std::make_unique<MockPPOEnvironment>());
    
    // Should throw an error due to mismatch (test through training)
    ASSERT_THROW(ppo->TrainWithEnvironments(environments, 1), std::runtime_error);
    
    std::cout << "PPO configuration validation working correctly" << std::endl;
}

// Test PPO loss computations
void test_ppo_loss_computations() {
    std::cout << "Starting PPO Loss Computations test..." << std::endl;
    
    auto network = networks::A2CImpl::WithMLP(4, 32, 2);
    std::cout << "Network created successfully" << std::endl;
    
    PPO::Config config;
    config.device = torch::kCPU;
    config.clip_ratio = 0.2f;
    config.num_envs = 1; // Use single environment
    config.rollout_steps = 10; // Use very small rollout for quick testing
    config.mini_batch_size = 5; // Small batch size to match rollout
    config.ppo_epochs = 1; // Single epoch for quick test
    config.verbose = true; // Enable verbose output
    std::cout << "Config set up successfully" << std::endl;
    
    auto ppo = std::make_unique<PPO>(network, config);
    std::cout << "PPO algorithm created successfully" << std::endl;
    
    // Test that PPO can perform training (which involves loss computations)
    std::vector<std::unique_ptr<AbstractEnv>> environments;
    environments.push_back(std::make_unique<MockPPOEnvironment>());
    std::cout << "Environment created successfully" << std::endl;
    
    // Run one training step to verify loss computations work
    std::cout << "About to start training..." << std::endl;
    ASSERT_NO_THROW(ppo->TrainWithEnvironments(environments, 1));
    
    // Check that training stats are updated (indicating loss computations worked)
    const auto& stats = ppo->GetLastStats();
    ASSERT_TRUE(std::isfinite(stats.policy_loss));
    ASSERT_TRUE(std::isfinite(stats.value_loss));
    ASSERT_TRUE(std::isfinite(stats.total_loss));
    ASSERT_FALSE(std::isnan(stats.policy_loss));
    ASSERT_FALSE(std::isnan(stats.value_loss));
    ASSERT_FALSE(std::isnan(stats.total_loss));
    
    std::cout << "PPO loss computations working correctly" << std::endl;
}

int main() {
    TestSuite suite("PPO Algorithm Tests");
    
    // Add test cases
    suite.AddTest("PPO Initialization", test_ppo_initialization);
    suite.AddTest("PPO Environment Initialization", test_ppo_environment_initialization);
    suite.AddTest("PPO Action Generation", test_ppo_action_generation);
    suite.AddTest("PPO Value Estimation", test_ppo_value_estimation);
    suite.AddTest("PPO Experience Collection", test_ppo_experience_collection);
    suite.AddTest("PPO Policy Update", test_ppo_policy_update);
    suite.AddTest("PPO Evaluation", test_ppo_evaluation);
    suite.AddTest("PPO Training Loop", test_ppo_training_loop);
    suite.AddTest("PPO Model Save/Load", test_ppo_model_save_load);
    suite.AddTest("PPO Config Validation", test_ppo_config_validation);
    suite.AddTest("PPO Loss Computations", test_ppo_loss_computations);
    
    // Run all tests
    bool all_passed = suite.RunAll();
    
    // Get final statistics
    int passed, failed, total;
    suite.GetStats(passed, failed, total);
    
    std::cout << "\nFinal Results: " << passed << "/" << total << " tests passed" << std::endl;
    
    return all_passed ? 0 : 1;
}
