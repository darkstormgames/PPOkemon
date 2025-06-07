#include <torch/torch.h>
#include <filesystem>
#include <memory>
#include "test_utils.h"
#include "../include/torch/training/callbacks/checkpoint_callback.h"
#include "../include/torch/training/callbacks/logging_callback.h"
#include "../include/torch/utils/logger.h"

namespace fs = std::filesystem;
using namespace PPOkemonTest;
using namespace training;

class CallbackTestFixture {
public:
    void SetUp() {
        test_dir_ = fs::temp_directory_path() / "ppokemon_callback_test";
        fs::create_directories(test_dir_);
    }
    
    void TearDown() {
        if (fs::exists(test_dir_)) {
            fs::remove_all(test_dir_);
        }
    }
    
    fs::path test_dir_;
};

void test_checkpoint_callback_basics() {
    CallbackTestFixture fixture;
    fixture.SetUp();
    
    // Create checkpoint callback config
    CheckpointCallback::Config config;
    config.checkpoint_dir = fixture.test_dir_ / "checkpoints";
    config.save_interval = 5;
    config.max_to_keep = 3;
    
    CheckpointCallback callback(config);
    
    // Test callback lifecycle
    ASSERT_NO_THROW(callback.OnTrainBegin());
    
    // Test step end callbacks
    for (int step = 1; step <= 10; ++step) {
        float reward = 100.0f + step;
        float loss = 1.0f / step;
        ASSERT_NO_THROW(callback.OnStepEnd(step, reward, loss));
    }
    
    ASSERT_NO_THROW(callback.OnTrainEnd());
    
    // Check that checkpoint directory was created
    ASSERT_TRUE(fs::exists(config.checkpoint_dir));
    
    fixture.TearDown();
}

void test_checkpoint_callback_best_model() {
    CallbackTestFixture fixture;
    fixture.SetUp();
    
    // Create checkpoint callback config
    CheckpointCallback::Config config;
    config.checkpoint_dir = fixture.test_dir_ / "checkpoints";
    config.save_interval = 1;  // Save every step for testing
    config.max_to_keep = 3;
    config.save_only_best = true;
    
    CheckpointCallback callback(config);
    
    // Set a mock save function
    callback.SetSaveModelFunction([](const std::string& path) -> bool {
        (void)path; // Mock save function that always succeeds
        return true;
    });
    
    callback.OnTrainBegin();
    
    // Simulate improving performance
    std::vector<float> rewards = {50.0f, 75.0f, 100.0f, 90.0f, 120.0f};
    
    for (size_t step = 1; step <= rewards.size(); ++step) {
        float reward = rewards[step - 1];
        float loss = 1.0f / step;
        ASSERT_NO_THROW(callback.OnStepEnd(static_cast<int64_t>(step), reward, loss));
    }
    
    // Best metric should be tracked
    ASSERT_TRUE(callback.GetBestMetric() > 0);
    
    fixture.TearDown();
}

void test_logging_callback_basics() {
    CallbackTestFixture fixture;
    fixture.SetUp();
    
    // Create logging callback config
    LoggingCallback::Config config;
    config.log_dir = fixture.test_dir_ / "logs";
    config.log_interval = 2;
    
    LoggingCallback callback(config);
    
    ASSERT_NO_THROW(callback.OnTrainBegin());
    
    // Test logging at intervals
    for (int step = 1; step <= 5; ++step) {
        float reward = 100.0f + step * 10;
        float loss = 1.0f / step;
        ASSERT_NO_THROW(callback.OnStepEnd(step, reward, loss));
    }
    
    ASSERT_NO_THROW(callback.OnTrainEnd());
    
    // Check that log directory was created
    ASSERT_TRUE(fs::exists(config.log_dir));
    
    fixture.TearDown();
}

void test_callbacks_with_empty_metrics() {
    CallbackTestFixture fixture;
    fixture.SetUp();
    
    // Create callback configs
    CheckpointCallback::Config checkpoint_config;
    checkpoint_config.checkpoint_dir = fixture.test_dir_ / "empty_checkpoints";
    
    LoggingCallback::Config logging_config;
    logging_config.log_dir = fixture.test_dir_ / "empty_logs";
    
    CheckpointCallback checkpoint_cb(checkpoint_config);
    LoggingCallback logging_cb(logging_config);
    
    // Test with minimal input
    ASSERT_NO_THROW(checkpoint_cb.OnTrainBegin());
    ASSERT_NO_THROW(checkpoint_cb.OnStepEnd(1, 0.0f, 0.0f));
    ASSERT_NO_THROW(checkpoint_cb.OnTrainEnd());
    
    ASSERT_NO_THROW(logging_cb.OnTrainBegin());
    ASSERT_NO_THROW(logging_cb.OnStepEnd(1, 0.0f, 0.0f));
    ASSERT_NO_THROW(logging_cb.OnTrainEnd());
    
    fixture.TearDown();
}

int main() {
    TestSuite suite("PPOkemon Training Callbacks Tests");
    
    suite.AddTest("Checkpoint Callback Basics", test_checkpoint_callback_basics);
    suite.AddTest("Checkpoint Callback Best Model", test_checkpoint_callback_best_model);
    suite.AddTest("Logging Callback Basics", test_logging_callback_basics);
    suite.AddTest("Callbacks with Empty Metrics", test_callbacks_with_empty_metrics);
    
    bool all_passed = suite.RunAll();
    
    int passed, failed, total;
    suite.GetStats(passed, failed, total);
    
    std::cout << "\nFinal Results: " << passed << "/" << total << " tests passed" << std::endl;
    
    return all_passed ? 0 : 1;
}
