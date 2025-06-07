#include <torch/torch.h>
#include <memory>
#include <vector>
#include <filesystem>
#include <fstream>
#include "test_utils.h"
#include "../include/torch/training/trainer.h"

namespace fs = std::filesystem;
using namespace PPOkemonTest;
using namespace training;

class TrainerTestFixture {
public:
    void SetUp() {
        test_dir_ = fs::temp_directory_path() / "ppokemon_trainer_test";
        fs::create_directories(test_dir_);
    }
    
    void TearDown() {
        if (fs::exists(test_dir_)) {
            fs::remove_all(test_dir_);
        }
    }
    
    fs::path test_dir_;
};

// Mock trainer for testing base functionality
class MockTrainer : public Trainer {
public:
    MockTrainer(const Trainer::Config& config)
        : Trainer(config), train_step_calls_(0) {}
    
    // Implement pure virtual functions
    void Train() override {
        for (int64_t step = 1; step <= config_.total_steps; ++step) {
            current_step_ = step;
            CollectExperience();
            float loss = UpdatePolicy();
            train_step_calls_++;
            
            // Simulate metrics
            float reward = 100.0f + step;
            current_reward_ = reward;
            
            // Use the proper logging mechanism
            std::unordered_map<std::string, float> metrics = {
                {"reward", reward},
                {"loss", loss}
            };
            LogMetrics(metrics);
        }
    }
    
    void Evaluate(int num_episodes = 10) override {
        (void)num_episodes; // Mock evaluation
        float eval_reward = 150.0f;
        
        // Call eval callbacks if any are registered
        for (auto& callback : eval_callbacks_) {
            callback(1, eval_reward);
        }
    }
    
    void SaveModel(const std::string& path) override {
        (void)path; // Mock save
    }
    
    void LoadModel(const std::string& path) override {
        (void)path; // Mock load
    }
    
    // Implement pure virtual training methods
    void CollectExperience() override {
        // Mock experience collection
    }
    
    float UpdatePolicy() override {
        // Mock policy update
        return 0.5f; // Mock loss
    }
    
    void ResetEnvironments() override {
        // Mock environment reset
    }
    
    int GetTrainStepCalls() const { return train_step_calls_; }
    
private:
    int train_step_calls_;
};

void test_trainer_basic_functionality() {
    TrainerTestFixture fixture;
    fixture.SetUp();
    
    // Create trainer config
    Trainer::Config config;
    config.total_steps = 5;
    config.log_dir = fixture.test_dir_ / "trainer_logs";
    config.checkpoint_dir = fixture.test_dir_ / "trainer_checkpoints";
    
    MockTrainer trainer(config);
    
    // Test training
    ASSERT_NO_THROW(trainer.Train());
    
    // Verify that training steps were called
    ASSERT_EQ(trainer.GetTrainStepCalls(), config.total_steps);
    
    fixture.TearDown();
}

void test_trainer_callback_integration() {
    TrainerTestFixture fixture;
    fixture.SetUp();
    
    // Create trainer config
    Trainer::Config config;
    config.total_steps = 3;
    config.log_dir = fixture.test_dir_ / "trainer_logs";
    config.checkpoint_dir = fixture.test_dir_ / "trainer_checkpoints";
    config.save_interval = 1;
    config.log_interval = 1;
    
    MockTrainer trainer(config);
    
    // Add step callback
    int step_callback_calls = 0;
    trainer.RegisterStepCallback([&step_callback_calls](int64_t step, float reward, float loss) {
        (void)step; (void)reward; (void)loss;
        step_callback_calls++;
    });
    
    // Add eval callback
    int eval_callback_calls = 0;
    trainer.RegisterEvalCallback([&eval_callback_calls](int64_t step, float eval_reward) {
        (void)step; (void)eval_reward;
        eval_callback_calls++;
    });
    
    ASSERT_NO_THROW(trainer.Train());
    ASSERT_NO_THROW(trainer.Evaluate());
    
    // Verify callbacks were called
    ASSERT_EQ(step_callback_calls, config.total_steps);
    ASSERT_EQ(eval_callback_calls, 1);
    
    fixture.TearDown();
}

void test_trainer_configuration() {
    Trainer::Config config;
    config.total_steps = 1000;
    config.batch_size = 64;
    config.learning_rate = 0.001f;
    config.weight_decay = 0.99f;  // Use weight_decay instead of discount_factor
    config.save_interval = 100;
    config.log_interval = 10;
    config.eval_interval = 500;
    
    MockTrainer trainer(config);
    
    // Test that trainer is properly initialized with config
    ASSERT_NO_THROW(trainer.Train());
}

void test_full_training_pipeline_integration() {
    TrainerTestFixture fixture;
    fixture.SetUp();
    
    // Create trainer config
    Trainer::Config config;
    config.total_steps = 5;
    config.log_dir = fixture.test_dir_ / "integration_logs";
    config.checkpoint_dir = fixture.test_dir_ / "integration_checkpoints";
    config.save_interval = 2;
    config.log_interval = 1;
    
    MockTrainer trainer(config);
    
    // Test full pipeline
    ASSERT_NO_THROW(trainer.Train());
    ASSERT_NO_THROW(trainer.Evaluate());
    
    // Verify training was executed
    ASSERT_EQ(trainer.GetTrainStepCalls(), config.total_steps);
    
    fixture.TearDown();
}

int main() {
    TestSuite suite("PPOkemon Training Trainer Tests");
    
    suite.AddTest("Basic Functionality", test_trainer_basic_functionality);
    suite.AddTest("Callback Integration", test_trainer_callback_integration);
    suite.AddTest("Configuration", test_trainer_configuration);
    suite.AddTest("Full Pipeline Integration", test_full_training_pipeline_integration);
    
    bool all_passed = suite.RunAll();
    
    int passed, failed, total;
    suite.GetStats(passed, failed, total);
    
    std::cout << "\nFinal Results: " << passed << "/" << total << " tests passed" << std::endl;
    
    return all_passed ? 0 : 1;
}
