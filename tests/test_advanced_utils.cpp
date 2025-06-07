/**
 * @file test_advanced_utils.cpp
 * @brief Integration test suite for advanced utility components
 * 
 * This file provides high-level integration tests that verify the utility
 * components work together correctly. For detailed component-specific tests, see:
 * - test_utils_seed.cpp: Seed manager tests
 * - test_utils_replay.cpp: Replay buffer tests
 * - test_utils_profiler.cpp: Profiler tests
 * - test_utils_scheduler.cpp: Learning rate scheduler tests
 * - test_utils_ml.cpp: Machine learning utilities tests
 */

#include <iostream>
#include <torch/torch.h>
#include <chrono>
#include <thread>
#include <filesystem>
#include "test_utils.h"

// Include all the utility headers
#include "../include/torch/utils/seed_manager.h"
#include "../include/torch/utils/replay_buffer.h"
#include "../include/torch/utils/profiler.h"
#include "../include/torch/utils/lr_scheduler.h"
#include "../include/torch/utils/logger.h"
#include "../include/torch/utils/checkpoint_manager.h"

using namespace PPOkemonTest;

// ============================================================================
// Integration Test Functions
// ============================================================================

void test_seed_manager_integration() {
    std::cout << "\n=== Testing Seed Manager Integration ===" << std::endl;
    
    // Test global seed setting affects all random operations
    utils::SeedManager::SetGlobalSeed(12345);
    
    // Test reproducibility across different components
    auto tensor1 = torch::randn({3, 3});
    auto seeds1 = utils::SeedManager::GenerateSeeds(42, 3);
    auto rand_float1 = utils::SeedManager::RandomFloat(0.0f, 1.0f);
    
    // Reset and test again
    utils::SeedManager::SetGlobalSeed(12345);
    auto tensor2 = torch::randn({3, 3});
    auto seeds2 = utils::SeedManager::GenerateSeeds(42, 3);
    auto rand_float2 = utils::SeedManager::RandomFloat(0.0f, 1.0f);
    
    bool reproducible = torch::allclose(tensor1, tensor2) && 
                       (seeds1 == seeds2) && 
                       (std::abs(rand_float1 - rand_float2) < 1e-6f);
    
    std::cout << "Cross-component reproducibility: " << (reproducible ? "✓" : "✗") << std::endl;
}

void test_replay_buffer_profiler_integration() {
    std::cout << "\n=== Testing Replay Buffer + Profiler Integration ===" << std::endl;
    
    auto& profiler = utils::Profiler::Instance();
    utils::ReplayBuffer buffer(1000);
    
    // Profile buffer operations
    {
        PROFILE_SCOPE("buffer_operations");
        
        for (int i = 0; i < 100; ++i) {
            auto state = torch::randn({4});
            auto action = torch::randint(0, 2, {1}).to(torch::kFloat32);
            auto reward_tensor = torch::randn({1});
            float reward = reward_tensor.item<float>();  // Convert tensor to float
            auto next_state = torch::randn({4});
            bool done = torch::randint(0, 2, {1}).item<bool>();
            
            buffer.Push(state, action, reward, next_state, done);
        }
        
        // Sample from buffer
        auto [states, actions, rewards, next_states, dones] = buffer.Sample(32);
    }
    
    double avg_time = profiler.GetAverageTime("buffer_operations");
    std::cout << "Buffer operations profiled: " << (avg_time > 0 ? "✓" : "✗") << std::endl;
    std::cout << "Average time: " << (avg_time * 1000.0) << " ms" << std::endl;
}

void test_scheduler_logger_integration() {
    std::cout << "\n=== Testing Scheduler + Logger Integration ===" << std::endl;
    
    // Create a simple model and optimizer
    torch::nn::Linear model(10, 1);
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.1));
    
    utils::LearningRateScheduler scheduler(optimizer, utils::SchedulerType::LINEAR_DECAY, 0.1f, 100);
    scheduler.SetLinearDecay(0.01f);
    
    utils::TrainingLogger logger("./test_integration_logs", "scheduler_test");
    
    // Simulate training with learning rate scheduling and logging
    for (int step = 0; step < 100; step += 10) {
        scheduler.Step(step);
        float current_lr = scheduler.GetLR();
        float loss = 1.0f / (1.0f + step * 0.01f);
        
        logger.LogScalar("learning_rate", current_lr, step);
        logger.LogScalar("loss", loss, step);
    }
    
    // Verify logging worked
    const auto& lr_data = logger.GetMetric("learning_rate");
    bool has_lr_data = lr_data.GetLatest() > 0;
    
    std::cout << "Scheduler-Logger integration: " << (has_lr_data ? "✓" : "✗") << std::endl;
    
    // Cleanup
    try {
        std::filesystem::remove_all("./test_integration_logs");
    } catch (...) {}
}

void test_full_utils_workflow() {
    std::cout << "\n=== Testing Full Utility Workflow ===" << std::endl;
    
    // Set up reproducible environment
    utils::SeedManager::SetGlobalSeed(54321);
    utils::SeedManager::EnableDeterministicMode();
    
    auto& profiler = utils::Profiler::Instance();
    
    {
        PROFILE_SCOPE("full_workflow");
        
        // Create replay buffer
        utils::ReplayBuffer buffer(500);
        
        // Create scheduler and logger
        torch::nn::Linear model(8, 4);
        torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.001));
        utils::LearningRateScheduler scheduler(optimizer, utils::SchedulerType::COSINE_ANNEALING, 0.001f, 50);
        scheduler.SetCosineAnnealing(0.0001f);
        
        utils::TrainingLogger logger("./test_full_workflow_logs", "full_test");
        
        // Simulate training workflow
        for (int episode = 0; episode < 50; ++episode) {
            // Generate episode data
            auto state = torch::randn({8});
            auto action = torch::randint(0, 4, {1}).to(torch::kFloat32);
            auto reward = utils::SeedManager::RandomFloat(-1.0f, 1.0f);
            auto next_state = torch::randn({8});
            bool done = torch::randint(0, 2, {1}).item<bool>();
            
            // Store in buffer (reward is already float)
            buffer.Push(state, action, reward, next_state, done);
            
            // Update learning rate
            scheduler.Step(episode);
            
            // Log metrics
            float current_lr = scheduler.GetLR();
            logger.LogScalar("learning_rate", current_lr, episode);
            logger.LogScalar("reward", reward, episode);
            logger.LogScalar("buffer_size", static_cast<float>(buffer.Size()), episode);
        }
        
        // Test buffer sampling
        if (buffer.Size() >= 32) {
            auto [states, actions, rewards, next_states, dones] = buffer.Sample(32);
            std::cout << "Buffer sampling: ✓ (sampled " << states.size() << " transitions)" << std::endl;
        }
    }
    
    // Check profiling results
    double workflow_time = profiler.GetAverageTime("full_workflow");
    std::cout << "Full workflow profiled: " << (workflow_time > 0 ? "✓" : "✗") << std::endl;
    
    // Cleanup
    try {
        std::filesystem::remove_all("./test_full_workflow_logs");
    } catch (...) {}
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    TestSuite suite("PPOkemon Advanced Utilities Integration Tests");
    
    std::cout << "=== PPOkemon Advanced Utilities Integration Tests ===" << std::endl;
    std::cout << "This suite tests high-level integration between utility components." << std::endl;
    std::cout << "For detailed component tests, see:" << std::endl;
    std::cout << "  - test_utils_seed.cpp" << std::endl;
    std::cout << "  - test_utils_replay.cpp" << std::endl;
    std::cout << "  - test_utils_profiler.cpp" << std::endl;
    std::cout << "  - test_utils_scheduler.cpp" << std::endl;
    std::cout << "  - test_utils_ml.cpp" << std::endl;
    std::cout << std::endl;
    
    // Add integration test cases
    suite.AddTest("Seed Manager Integration", test_seed_manager_integration);
    suite.AddTest("Replay Buffer + Profiler Integration", test_replay_buffer_profiler_integration);
    suite.AddTest("Scheduler + Logger Integration", test_scheduler_logger_integration);
    suite.AddTest("Full Utilities Workflow", test_full_utils_workflow);
    
    // Run all tests
    bool all_passed = suite.RunAll();
    
    // Get final statistics
    int passed, failed, total;
    suite.GetStats(passed, failed, total);
    
    std::cout << "\nFinal Results: " << passed << "/" << total << " integration tests passed" << std::endl;
    
    return all_passed ? 0 : 1;
}
