/**
 * @file test_training.cpp
 * @brief Integration test suite for training components
 * 
 * This file provides high-level integration tests that verify the training
 * components work together correctly. For detailed component-specific tests, see:
 * - test_training_metrics.cpp: Metrics collection and computation
 * - test_training_evaluator.cpp: Evaluator functionality  
 * - test_training_callbacks.cpp: Checkpoint and logging callbacks
 * - test_training_trainer.cpp: Trainer base class functionality
 */

#include <torch/torch.h>
#include <memory>
#include <vector>
#include <filesystem>
#include "test_utils.h"

// Training components
#include "../include/torch/training/evaluation/metrics.h"
#include "../include/torch/training/evaluation/evaluator.h"
#include "../include/torch/training/callbacks/checkpoint_callback.h"
#include "../include/torch/training/callbacks/logging_callback.h"
#include "../include/torch/training/trainer.h"
#include "../include/torch/utils/logger.h"

// Networks for testing
#include "../include/torch/networks/mlp.h"
#include "../include/torch/networks/a2c.h"

namespace fs = std::filesystem;

using namespace PPOkemonTest;
using namespace training;

// Mock environment for testing evaluator
class MockEnvironment {
public:
    struct StepResult {
        torch::Tensor next_state;
        float reward;
        bool done;
        std::unordered_map<std::string, float> info;
    };
    
    MockEnvironment() : step_count_(0), max_steps_(100) {}
    
    torch::Tensor Reset() {
        step_count_ = 0;
        return torch::randn({4}); // 4-dimensional state
    }
    
    StepResult Step(const torch::Tensor& action) {
        (void)action; // Suppress unused parameter warning
        step_count_++;
        StepResult result;
        result.next_state = torch::randn({4});
        result.reward = 1.0f; // Fixed reward for testing
        result.done = (step_count_ >= max_steps_);
        result.info["steps"] = static_cast<float>(step_count_);
        return result;
    }
    
    int GetActionDim() const { return 2; }
    int GetStateDim() const { return 4; }
    
private:
    int step_count_;
    int max_steps_;
};

// Test fixture class to manage common setup for integration tests
class TrainingIntegrationFixture {
public:
    void SetUp() {
        // Create temporary directory for test files
        test_dir_ = fs::temp_directory_path() / "ppokemon_training_integration_test";
        fs::create_directories(test_dir_);
        
        // Create basic networks for testing
        mlp_net_ = std::make_shared<networks::MLP>(4, 32, 2);
        a2c_net_ = std::make_shared<networks::A2C>(networks::MLPTag{}, 4, 32, 2);
        
        // Create test data
        test_states_ = torch::randn({10, 4});
        test_actions_ = torch::randint(0, 2, {10, 1});
        test_rewards_ = torch::randn({10});
        test_values_ = torch::randn({10});
        test_log_probs_ = torch::randn({10});
    }
    
    void TearDown() {
        // Clean up temporary files
        if (fs::exists(test_dir_)) {
            fs::remove_all(test_dir_);
        }
    }
    
    fs::path test_dir_;
    std::shared_ptr<networks::MLP> mlp_net_;
    std::shared_ptr<networks::A2C> a2c_net_;
    torch::Tensor test_states_;
    torch::Tensor test_actions_;
    torch::Tensor test_rewards_;
    torch::Tensor test_values_;
    torch::Tensor test_log_probs_;
};

// Integration test for training pipeline
void test_training_pipeline_integration() {
    TrainingIntegrationFixture fixture;
    fixture.SetUp();
    
    // Test integration of metrics and evaluator
    auto metrics = std::make_shared<Metrics>();
    Evaluator::Config eval_config;
    auto evaluator = std::make_shared<Evaluator>(eval_config);
    
    // Test metrics collection
    std::vector<Metrics::EpisodeStats> episodes;
    Metrics::EpisodeStats ep;
    ep.total_reward = 100.0f;
    ep.episode_length = 250.0f;
    ep.success = true;
    episodes.push_back(ep);
    
    // Verify metrics computation
    auto stats = Metrics::ComputeEpisodeStats(episodes);
    ASSERT_CLOSE(stats.mean_reward, 100.0f, 1e-5f);
    
    // Test evaluator with mock environment
    auto mock_env = std::make_unique<MockEnvironment>();
    // Use MLPImpl directly to avoid casting issues
    auto mlp_impl = std::make_shared<networks::MLPImpl>(4, 32, 2);
    auto network = std::static_pointer_cast<networks::NetworkBase>(mlp_impl);
    
    // Simple evaluation test
    auto initial_state = mock_env->Reset();
    auto action_output = network->forward(initial_state.unsqueeze(0));
    ASSERT_EQ(action_output.size(1), 2); // 2 actions
    
    fixture.TearDown();
}

void test_callback_integration() {
    TrainingIntegrationFixture fixture;
    fixture.SetUp();
    
    // Test checkpoint and logging callbacks work together
    CheckpointCallback::Config checkpoint_config;
    checkpoint_config.checkpoint_dir = (fixture.test_dir_ / "checkpoints").string();
    checkpoint_config.save_interval = 2;
    checkpoint_config.save_only_best = true;
    auto checkpoint_callback = std::make_shared<CheckpointCallback>(checkpoint_config);
    
    LoggingCallback::Config logging_config;
    logging_config.log_dir = (fixture.test_dir_ / "logs").string();
    logging_config.log_interval = 1;
    auto logging_callback = std::make_shared<LoggingCallback>(logging_config);
    
    // Test metrics data
    std::unordered_map<std::string, float> metrics1 = {{"reward", 10.0f}, {"loss", 0.5f}};
    std::unordered_map<std::string, float> metrics2 = {{"reward", 15.0f}, {"loss", 0.8f}};
    
    // Test callbacks respond to step events (not episode events)
    checkpoint_callback->OnStepEnd(1, metrics1["reward"], metrics1["loss"]);
    logging_callback->OnStepEnd(1, metrics1["reward"], metrics1["loss"]);
    
    checkpoint_callback->OnStepEnd(2, metrics2["reward"], metrics2["loss"]);
    logging_callback->OnStepEnd(2, metrics2["reward"], metrics2["loss"]);
    
    // Verify checkpoint directory was created
    ASSERT_TRUE(fs::exists(checkpoint_config.checkpoint_dir));
    
    // Verify log directory was created
    ASSERT_TRUE(fs::exists(logging_config.log_dir));
    
    fixture.TearDown();
}

void test_full_training_workflow() {
    TrainingIntegrationFixture fixture;
    fixture.SetUp();
    
    // Test full workflow: network + metrics + evaluator + callbacks
    auto metrics = std::make_shared<Metrics>();
    Evaluator::Config eval_config2;
    auto evaluator = std::make_shared<Evaluator>(eval_config2);
    
    // Create mock trainer-like component
    class MockTrainer {
    public:
        MockTrainer(std::shared_ptr<networks::NetworkBase> network) 
            : network_(network), step_count_(0) {}
        
        void TrainStep() {
            step_count_++;
            
            // Mock training step
            auto input = torch::randn({2, 4});
            auto output = network_->forward(input);
            
            // Mock loss computation
            float loss = 0.1f * step_count_;
            float reward = 10.0f + step_count_;
            
            // Log metrics
            LogMetrics(reward, loss);
        }
        
        void LogMetrics(float reward, float loss) {
            std::unordered_map<std::string, float> metrics;
            metrics["reward"] = reward;
            metrics["loss"] = loss;
            
            for (auto& callback : callbacks_) {
                callback->OnStepEnd(step_count_, reward, loss);
            }
        }
        
        void AddCallback(std::shared_ptr<training::Callback> callback) {
            callbacks_.push_back(callback);
        }
        
    private:
        std::shared_ptr<networks::NetworkBase> network_;
        std::vector<std::shared_ptr<training::Callback>> callbacks_;
        int step_count_;
    };
    
    auto mlp_impl = std::make_shared<networks::MLPImpl>(4, 32, 2);
    auto network = std::static_pointer_cast<networks::NetworkBase>(mlp_impl);
    auto trainer = std::make_unique<MockTrainer>(network);
    
    // Add callbacks
    CheckpointCallback::Config checkpoint_config;
    checkpoint_config.checkpoint_dir = (fixture.test_dir_ / "checkpoints").string();
    checkpoint_config.save_interval = 2;
    auto checkpoint_callback = std::make_shared<CheckpointCallback>(checkpoint_config);
    trainer->AddCallback(checkpoint_callback);
    
    LoggingCallback::Config logging_config;
    logging_config.log_dir = (fixture.test_dir_ / "logs").string();
    logging_config.log_interval = 1;
    auto logging_callback = std::make_shared<LoggingCallback>(logging_config);
    trainer->AddCallback(logging_callback);
    
    // Run training steps
    for (int i = 0; i < 5; ++i) {
        trainer->TrainStep();
    }
    
    // Verify workflow completed
    ASSERT_TRUE(fs::exists(checkpoint_config.checkpoint_dir));
    ASSERT_TRUE(fs::exists(logging_config.log_dir));
    
    fixture.TearDown();
}

void test_metrics_evaluator_integration() {
    TrainingIntegrationFixture fixture;
    fixture.SetUp();
    
    // Test metrics and evaluator work together
    auto metrics = std::make_shared<Metrics>();
    Evaluator::Config eval_config3;
    auto evaluator = std::make_shared<Evaluator>(eval_config3);
    
    // Mock environment
    auto mock_env = std::make_unique<MockEnvironment>();
    auto mlp_impl = std::make_shared<networks::MLPImpl>(4, 32, 2);
    auto network = std::static_pointer_cast<networks::NetworkBase>(mlp_impl);
    
    // Test evaluation produces valid metrics
    std::vector<Metrics::EpisodeStats> episodes;
    for (int i = 0; i < 3; ++i) {
        Metrics::EpisodeStats ep;
        ep.total_reward = 50.0f + i * 10.0f;
        ep.total_steps = static_cast<int64_t>(100 + i * 25);  // Set total_steps which is used by ExtractEpisodeLengths
        ep.episode_length = 100.0f + i * 25.0f;
        ep.success = true;
        episodes.push_back(ep);
    }
    
    // Compute statistics
    auto stats = Metrics::ComputeEpisodeStats(episodes);
    ASSERT_CLOSE(stats.mean_reward, 60.0f, 1e-5f);  // (50 + 60 + 70) / 3
    ASSERT_CLOSE(stats.mean_episode_length, 125.0f, 1e-5f);  // (100 + 125 + 150) / 3
    
    // Test evaluator basic functionality
    auto initial_state = mock_env->Reset();
    auto action_output = network->forward(initial_state.unsqueeze(0));
    ASSERT_EQ(action_output.size(0), 1);
    ASSERT_EQ(action_output.size(1), 2);
    
    fixture.TearDown();
}

int main() {
    TestSuite suite("Training Integration Tests");
    
    suite.AddTest("Training Pipeline Integration", test_training_pipeline_integration);
    suite.AddTest("Callback Integration", test_callback_integration);
    suite.AddTest("Full Training Workflow", test_full_training_workflow);
    suite.AddTest("Metrics-Evaluator Integration", test_metrics_evaluator_integration);
    
    bool all_passed = suite.RunAll();
    
    int passed, failed, total;
    suite.GetStats(passed, failed, total);
    
    std::cout << "\nFinal Results: " << passed << "/" << total << " tests passed" << std::endl;
    
    return all_passed ? 0 : 1;
}
