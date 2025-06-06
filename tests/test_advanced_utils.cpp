#include <iostream>
#include <torch/torch.h>
#include <chrono>
#include <thread>
#include <filesystem>

// Include all the new utility headers
#include "torch/utils/seed_manager.h"
#include "torch/utils/replay_buffer.h"
#include "torch/utils/profiler.h"
#include "torch/utils/lr_scheduler.h"
#include "torch/utils/logger.h"
#include "torch/utils/checkpoint_manager.h"

void test_seed_manager() {
    std::cout << "\n=== Testing Seed Manager ===" << std::endl;
    
    // Test global seed setting
    utils::SeedManager::SetGlobalSeed(12345);
    
    // Test reproducibility
    auto seeds1 = utils::SeedManager::GenerateSeeds(42, 5);
    auto seeds2 = utils::SeedManager::GenerateSeeds(42, 5);
    
    bool reproducible = true;
    for (size_t i = 0; i < seeds1.size(); ++i) {
        if (seeds1[i] != seeds2[i]) {
            reproducible = false;
            break;
        }
    }
    
    std::cout << "Current seed: " << utils::SeedManager::GetCurrentSeed() << std::endl;
    std::cout << "Generated 5 seeds reproducibly: " << (reproducible ? "✓" : "✗") << std::endl;
    
    // Test random utilities
    float rand_float = utils::SeedManager::RandomFloat(0.0f, 1.0f);
    int rand_int = utils::SeedManager::RandomInt(1, 100);
    bool rand_bool = utils::SeedManager::RandomBool(0.7f);
    
    std::cout << "Random float [0,1]: " << rand_float << std::endl;
    std::cout << "Random int [1,100]: " << rand_int << std::endl;
    std::cout << "Random bool (p=0.7): " << (rand_bool ? "true" : "false") << std::endl;
    
    // Test deterministic mode
    utils::SeedManager::EnableDeterministicMode();
    std::cout << "Deterministic mode enabled" << std::endl;
    
    // Test save/load state
    try {
        utils::SeedManager::SaveRandomState("./test_seed_state");
        utils::SeedManager::SetGlobalSeed(999);
        utils::SeedManager::LoadRandomState("./test_seed_state");
        std::cout << "Seed state save/load: ✓" << std::endl;
        
        // Cleanup
        std::filesystem::remove("./test_seed_state");
        std::filesystem::remove("./test_seed_state.names");
    } catch (const std::exception& e) {
        std::cout << "Seed state save/load failed: " << e.what() << std::endl;
    }
    
    std::cout << "Seed Manager test completed!" << std::endl;
}

void test_replay_buffer() {
    std::cout << "\n=== Testing Replay Buffer ===" << std::endl;
    
    // Create buffer
    utils::ReplayBuffer buffer(1000, 42);
    
    // Add some experiences
    for (int i = 0; i < 50; ++i) {
        torch::Tensor state = torch::randn({4});
        torch::Tensor action = torch::randint(0, 2, {1});
        float reward = static_cast<float>(i) * 0.1f;
        torch::Tensor next_state = torch::randn({4});
        bool done = (i % 10 == 9);
        
        buffer.Push(state, action, reward, next_state, done);
    }
    
    std::cout << "Buffer size: " << buffer.Size() << "//" << buffer.Capacity() << std::endl;
    std::cout << "Buffer is full: " << (buffer.IsFull() ? "true" : "false") << std::endl;
    
    // Test sampling
    auto batch = buffer.Sample(32);
    std::cout << "Sampled batch size: " << batch.size() << std::endl;
    std::cout << "Sample state shape: [" << batch.states[0].size(0) << "]" << std::endl;
    
    // Test sequential sampling
    auto seq_batch = buffer.SampleSequential(8, 4);
    std::cout << "Sequential batch size: " << seq_batch.size() << std::endl;
    
    // Test prioritized sampling
    std::vector<size_t> indices = {0, 1, 2, 3, 4};
    std::vector<float> priorities = {1.0f, 2.0f, 0.5f, 3.0f, 1.5f};
    buffer.UpdatePriorities(indices, priorities);
    
    auto prio_batch = buffer.SamplePrioritized(16);
    std::cout << "Prioritized batch size: " << prio_batch.size() << std::endl;
    
    // Test save/load
    try {
        buffer.Save("./test_buffer");
        utils::ReplayBuffer loaded_buffer(1000);
        loaded_buffer.Load("./test_buffer");
        std::cout << "Buffer save/load: ✓ (loaded size: " << loaded_buffer.Size() << ")" << std::endl;
        
        // Cleanup
        std::filesystem::remove("./test_buffer.tensors");
        std::filesystem::remove("./test_buffer.meta");
    } catch (const std::exception& e) {
        std::cout << "Buffer save/load failed: " << e.what() << std::endl;
    }
    
    std::cout << "Replay Buffer test completed!" << std::endl;
}

void test_profiler() {
    std::cout << "\n=== Testing Profiler ===" << std::endl;
    
    auto& profiler = utils::Profiler::Instance();
    
    // Test manual timing
    profiler.StartTimer("manual_test");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    profiler.StopTimer("manual_test");
    
    // Test scoped timing
    {
        PROFILE_SCOPE("scoped_test");
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    
    // Test function profiling
    auto do_work = []() {
        PROFILE_FUNCTION();
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    };
    
    // Run function multiple times
    for (int i = 0; i < 3; ++i) {
        do_work();
    }
    
    // Test direct time recording
    profiler.RecordTime("direct_record", 0.020); // 20ms
    
    // Print statistics
    std::cout << "Manual test average time: " << (profiler.GetAverageTime("manual_test") * 1000.0) << " ms" << std::endl;
    std::cout << "Scoped test call count: " << profiler.GetCallCount("scoped_test") << std::endl;
    std::cout << "Function test total time: " << (profiler.GetTotalTime("do_work") * 1000.0) << " ms" << std::endl;
    
    // Print full report
    profiler.PrintReport();
    
    // Test save report
    try {
        profiler.SaveReport("./profiler_report.txt");
        std::cout << "Profiler report saved to file: ✓" << std::endl;
        std::filesystem::remove("./profiler_report.txt");
    } catch (const std::exception& e) {
        std::cout << "Profiler save failed: " << e.what() << std::endl;
    }
    
    std::cout << "Profiler test completed!" << std::endl;
}

void test_lr_scheduler() {
    std::cout << "\n=== Testing Learning Rate Scheduler ===" << std::endl;
    
    // Create a simple model and optimizer
    torch::nn::Linear model(10, 1);
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.1));
    
    // Test different scheduler types
    {
        utils::LearningRateScheduler scheduler(optimizer, utils::SchedulerType::LINEAR_DECAY, 0.1f, 1000);
        scheduler.SetLinearDecay(0.01f);
        
        std::cout << "Initial LR: " << scheduler.GetLR() << std::endl;
        
        scheduler.Step(250);
        std::cout << "LR at step 250: " << scheduler.GetLR() << std::endl;
        
        scheduler.Step(500);
        std::cout << "LR at step 500: " << scheduler.GetLR() << std::endl;
        
        scheduler.Step(750);
        std::cout << "LR at step 750: " << scheduler.GetLR() << std::endl;
        
        scheduler.Step(1000);
        std::cout << "LR at step 1000 (final): " << scheduler.GetLR() << std::endl;
    }
    
    // Test cosine annealing
    {
        utils::LearningRateScheduler cos_scheduler(optimizer, utils::SchedulerType::COSINE_ANNEALING, 0.1f, 100);
        cos_scheduler.SetCosineAnnealing(0.001f);
        
        cos_scheduler.Step(25);
        std::cout << "Cosine LR at step 25/100: " << cos_scheduler.GetLR() << std::endl;
        
        cos_scheduler.Step(50);
        std::cout << "Cosine LR at step 50/100: " << cos_scheduler.GetLR() << std::endl;
        
        cos_scheduler.Step(100);
        std::cout << "Cosine LR at step 100/100: " << cos_scheduler.GetLR() << std::endl;
    }
    
    // Test custom scheduler
    {
        utils::LearningRateScheduler custom_scheduler(optimizer, utils::SchedulerType::CUSTOM, 0.1f, 1000);
        custom_scheduler.SetCustomScheduler([](int64_t step, int64_t total_steps, float initial_lr) {
            // Custom exponential decay
            float decay_rate = 0.95f;
            float decay_steps = 100.0f;
            return initial_lr * std::pow(decay_rate, step / decay_steps);
        });
        
        custom_scheduler.Step(200);
        std::cout << "Custom scheduler LR at step 200: " << custom_scheduler.GetLR() << std::endl;
    }
    
    std::cout << "Learning Rate Scheduler test completed!" << std::endl;
}

void test_logger() {
    std::cout << "\n=== Testing Training Logger ===" << std::endl;
    
    // Create logger
    utils::TrainingLogger logger("./test_logs", "test_experiment");
    
    // Log some metrics
    for (int step = 0; step < 100; step += 10) {
        float loss = 1.0f / (1.0f + step * 0.01f); // Decreasing loss
        float reward = step * 0.1f; // Increasing reward
        float lr = 0.001f * std::exp(-step * 0.01f); // Decaying learning rate
        
        logger.LogScalar("loss", loss, step);
        logger.LogScalar("reward", reward, step);
        logger.LogScalar("learning_rate", lr, step);
    }
    
    // Log multiple scalars at once
    std::unordered_map<std::string, float> metrics = {
        {"accuracy", 0.95f},
        {"precision", 0.92f},
        {"recall", 0.88f}
    };
    logger.LogScalars(metrics, 100);
    
    // Log text messages
    logger.LogText("Training started");
    logger.LogText("checkpoint", "Model checkpoint saved", 50);
    
    // Log configuration
    std::unordered_map<std::string, std::string> config = {
        {"batch_size", "32"},
        {"learning_rate", "0.001"},
        {"optimizer", "Adam"}
    };
    logger.LogConfig(config);
    
    // Test metrics access
    const auto& loss_data = logger.GetMetric("loss");
    std::cout << "Loss latest value: " << loss_data.GetLatest() << std::endl;
    std::cout << "Loss average (last 5): " << loss_data.GetAverage(5) << std::endl;
    
    // Print summary
    logger.PrintSummary(50);
    logger.PrintProgress(100, 1000);
    
    // Save reports
    try {
        logger.SaveToCSV("./test_metrics.csv");
        logger.SaveToJSON("./test_metrics.json");
        std::cout << "Logger save to CSV/JSON: ✓" << std::endl;
        
        // Cleanup
        std::filesystem::remove_all("./test_logs");
        std::filesystem::remove("./test_metrics.csv");
        std::filesystem::remove("./test_metrics.json");
    } catch (const std::exception& e) {
        std::cout << "Logger save failed: " << e.what() << std::endl;
    }
    
    std::cout << "Training Logger test completed!" << std::endl;
}

void test_checkpoint_manager() {
    std::cout << "\n=== Testing Checkpoint Manager ===" << std::endl;
    
    // Create a simple custom module for testing
    struct TestNet : torch::nn::Module {
        TestNet() {
            fc1 = register_module("fc1", torch::nn::Linear(10, 64));
            fc2 = register_module("fc2", torch::nn::Linear(64, 1));
        }
        
        torch::Tensor forward(torch::Tensor x) {
            x = torch::relu(fc1->forward(x));
            return fc2->forward(x);
        }
        
        torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    };
    
    auto model = std::make_shared<TestNet>();
    auto module_ptr = std::static_pointer_cast<torch::nn::Module>(model);
    torch::optim::Adam optimizer(module_ptr->parameters(), torch::optim::AdamOptions(0.001));
    
    // Create checkpoint manager
    utils::CheckpointManager ckpt_manager("./test_checkpoints", 3, "reward");
    
    // Save multiple checkpoints
    std::vector<float> rewards = {10.5f, 15.2f, 12.8f, 18.9f, 14.1f};
    std::vector<std::string> saved_paths;
    
    for (size_t i = 0; i < rewards.size(); ++i) {
        int64_t step = (i + 1) * 100;
        float reward = rewards[i];
        
        std::unordered_map<std::string, float> extra_metrics = {
            {"loss", 1.0f / (1.0f + i)},
            {"accuracy", 0.8f + i * 0.02f}
        };
        
        std::string path = ckpt_manager.SaveCheckpoint(module_ptr, optimizer, step, reward, extra_metrics);
        saved_paths.push_back(path);
        
        std::cout << "Saved checkpoint at step " << step << " with reward " << reward << std::endl;
    }
    
    std::cout << "Has checkpoints: " << (ckpt_manager.HasCheckpoints() ? "true" : "false") << std::endl;
    std::cout << "Latest checkpoint: " << ckpt_manager.GetLatestCheckpointPath() << std::endl;
    std::cout << "Best checkpoint: " << ckpt_manager.GetBestCheckpointPath() << std::endl;
    
    // Print checkpoint statistics
    ckpt_manager.PrintCheckpointStats();
    
    // Test loading latest checkpoint
    {
        auto test_model = std::make_shared<TestNet>();
        auto test_module_ptr = std::static_pointer_cast<torch::nn::Module>(test_model);
        torch::optim::Adam test_optimizer(test_module_ptr->parameters(), torch::optim::AdamOptions(0.001));
        
        int64_t loaded_step;
        float loaded_reward;
        
        bool success = ckpt_manager.LoadLatestCheckpoint(test_module_ptr, test_optimizer, loaded_step, loaded_reward);
        std::cout << "Load latest checkpoint: " << (success ? "✓" : "✗") << std::endl;
        if (success) {
            std::cout << "Loaded step: " << loaded_step << ", reward: " << loaded_reward << std::endl;
        }
    }
    
    // Test loading best checkpoint
    {
        auto test_model = std::make_shared<TestNet>();
        auto test_module_ptr = std::static_pointer_cast<torch::nn::Module>(test_model);
        torch::optim::Adam test_optimizer(test_module_ptr->parameters(), torch::optim::AdamOptions(0.001));
        
        int64_t loaded_step;
        float loaded_reward;
        
        bool success = ckpt_manager.LoadBestCheckpoint(test_module_ptr, test_optimizer, loaded_step, loaded_reward);
        std::cout << "Load best checkpoint: " << (success ? "✓" : "✗") << std::endl;
        if (success) {
            std::cout << "Best step: " << loaded_step << ", best reward: " << loaded_reward << std::endl;
        }
    }
    
    // List all checkpoints
    auto checkpoint_list = ckpt_manager.ListCheckpoints();
    std::cout << "Total checkpoints: " << checkpoint_list.size() << std::endl;
    
    // Cleanup
    try {
        ckpt_manager.DeleteAllCheckpoints();
        std::filesystem::remove_all("./test_checkpoints");
        std::cout << "Checkpoint cleanup: ✓" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Checkpoint cleanup failed: " << e.what() << std::endl;
    }
    
    std::cout << "Checkpoint Manager test completed!" << std::endl;
}

int main() {
    std::cout << "Testing PPOkemon Advanced ML Utilities" << std::endl;
    std::cout << "======================================" << std::endl;
    
    try {
        test_seed_manager();
        test_replay_buffer();
        test_profiler();
        test_lr_scheduler();
        test_logger();
        test_checkpoint_manager();
        
        std::cout << "\n🎉 All advanced utility tests completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
