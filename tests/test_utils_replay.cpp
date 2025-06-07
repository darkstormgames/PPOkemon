#include <iostream>
#include <filesystem>
#include <torch/torch.h>
#include "torch/utils/replay_buffer.h"
#include "test_utils.h"

using namespace PPOkemonTest;

void test_replay_buffer_basic() {
    utils::ReplayBuffer buffer(1000, 42);
    
    ASSERT_EQ(buffer.Size(), 0);
    ASSERT_EQ(buffer.Capacity(), 1000);
    ASSERT_FALSE(buffer.IsFull());
}

void test_replay_buffer_push_sample() {
    utils::ReplayBuffer buffer(100, 42);
    
    // Add some experiences
    for (int i = 0; i < 50; ++i) {
        torch::Tensor state = torch::randn({4});
        torch::Tensor action = torch::randint(0, 2, {1});
        float reward = static_cast<float>(i) * 0.1f;
        torch::Tensor next_state = torch::randn({4});
        bool done = (i % 10 == 9);
        
        buffer.Push(state, action, reward, next_state, done);
    }
    
    ASSERT_EQ(buffer.Size(), 50);
    ASSERT_FALSE(buffer.IsFull());
    
    // Test sampling
    auto batch = buffer.Sample(32);
    ASSERT_EQ(batch.size(), 32);
    ASSERT_EQ(batch.states[0].size(0), 4);
}

void test_replay_buffer_sequential_sampling() {
    utils::ReplayBuffer buffer(100, 42);
    
    // Fill buffer with enough data for sequential sampling
    for (int i = 0; i < 50; ++i) {
        torch::Tensor state = torch::randn({4});
        torch::Tensor action = torch::randint(0, 2, {1});
        float reward = static_cast<float>(i);
        torch::Tensor next_state = torch::randn({4});
        bool done = false;
        
        buffer.Push(state, action, reward, next_state, done);
    }
    
    // Test sequential sampling - sample 5 sequences of length 4 (total 20 transitions)
    auto seq_batch = buffer.SampleSequential(5, 4);
    ASSERT_EQ(seq_batch.size(), 20);
}

void test_replay_buffer_prioritized_sampling() {
    utils::ReplayBuffer buffer(100, 42);
    
    // Add experiences
    for (int i = 0; i < 10; ++i) {
        torch::Tensor state = torch::randn({4});
        torch::Tensor action = torch::randint(0, 2, {1});
        float reward = static_cast<float>(i);
        torch::Tensor next_state = torch::randn({4});
        bool done = false;
        
        buffer.Push(state, action, reward, next_state, done);
    }
    
    // Set priorities
    std::vector<size_t> indices = {0, 1, 2, 3, 4};
    std::vector<float> priorities = {1.0f, 2.0f, 0.5f, 3.0f, 1.5f};
    buffer.UpdatePriorities(indices, priorities);
    
    // Test prioritized sampling
    auto prio_batch = buffer.SamplePrioritized(5);
    ASSERT_EQ(prio_batch.size(), 5);
}

void test_replay_buffer_save_load() {
    utils::ReplayBuffer buffer(100, 42);
    
    // Add some data
    for (int i = 0; i < 10; ++i) {
        torch::Tensor state = torch::randn({4});
        torch::Tensor action = torch::randint(0, 2, {1});
        float reward = static_cast<float>(i);
        torch::Tensor next_state = torch::randn({4});
        bool done = false;
        
        buffer.Push(state, action, reward, next_state, done);
    }
    
    // Save buffer
    std::string save_path = "./test_buffer";
    ASSERT_NO_THROW(buffer.Save(save_path));
    
    // Load into new buffer
    utils::ReplayBuffer loaded_buffer(100);
    ASSERT_NO_THROW(loaded_buffer.Load(save_path));
    ASSERT_EQ(loaded_buffer.Size(), buffer.Size());
    
    // Cleanup
    std::filesystem::remove(save_path + ".tensors");
    std::filesystem::remove(save_path + ".meta");
}

int main() {
    TestSuite suite("PPOkemon Replay Buffer Tests");
    
    suite.AddTest("Basic Operations", test_replay_buffer_basic);
    suite.AddTest("Push and Sample", test_replay_buffer_push_sample);
    suite.AddTest("Sequential Sampling", test_replay_buffer_sequential_sampling);
    suite.AddTest("Prioritized Sampling", test_replay_buffer_prioritized_sampling);
    suite.AddTest("Save and Load", test_replay_buffer_save_load);
    
    bool all_passed = suite.RunAll();
    
    int passed, failed, total;
    suite.GetStats(passed, failed, total);
    
    std::cout << "\nFinal Results: " << passed << "/" << total << " tests passed" << std::endl;
    
    return all_passed ? 0 : 1;
}