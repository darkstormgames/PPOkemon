#pragma once

#include <torch/torch.h>
#include <vector>
#include <random>
#include <memory>

namespace utils {

struct Experience {
    torch::Tensor state;
    torch::Tensor action;
    float reward;
    torch::Tensor next_state;
    bool done;
    
    Experience(torch::Tensor s, torch::Tensor a, float r, torch::Tensor ns, bool d)
        : state(s), action(a), reward(r), next_state(ns), done(d) {}
};

struct Transition {
    std::vector<torch::Tensor> states;
    std::vector<torch::Tensor> actions;
    std::vector<float> rewards;
    std::vector<torch::Tensor> next_states;
    std::vector<bool> dones;
    
    void clear() {
        states.clear();
        actions.clear();
        rewards.clear();
        next_states.clear();
        dones.clear();
    }
    
    size_t size() const { return states.size(); }
};

class ReplayBuffer {
public:
    ReplayBuffer(size_t capacity, unsigned int seed = 42);
    ~ReplayBuffer() = default;
    
    // Add single experience
    void Push(const torch::Tensor& state, const torch::Tensor& action, 
              float reward, const torch::Tensor& next_state, bool done);
    
    // Add batch of experiences
    void PushBatch(const std::vector<Experience>& experiences);
    
    // Sample random batch
    Transition Sample(size_t batch_size);
    
    // Sample sequential batch (for RNNs)
    Transition SampleSequential(size_t batch_size, size_t sequence_length);
    
    // Sample prioritized batch (if priorities are set)
    Transition SamplePrioritized(size_t batch_size, float alpha = 0.6f, float beta = 0.4f);
    
    // Update priorities for prioritized replay
    void UpdatePriorities(const std::vector<size_t>& indices, const std::vector<float>& priorities);
    
    // Utility functions
    size_t Size() const { return current_size_; }
    size_t Capacity() const { return capacity_; }
    bool IsFull() const { return current_size_ == capacity_; }
    void Clear();
    
    // Save/Load buffer
    void Save(const std::string& path) const;
    void Load(const std::string& path);

private:
    size_t capacity_;
    size_t current_size_;
    size_t position_;
    
    // Storage
    std::vector<torch::Tensor> states_;
    std::vector<torch::Tensor> actions_;
    std::vector<float> rewards_;
    std::vector<torch::Tensor> next_states_;
    std::vector<bool> dones_;
    
    // Prioritized replay
    std::vector<float> priorities_;
    bool use_priorities_;
    
    // Random number generation
    std::mt19937 rng_;
    std::uniform_real_distribution<float> uniform_dist_;
    
    // Helper methods
    std::vector<size_t> SampleIndices(size_t batch_size);
    std::vector<size_t> SamplePrioritizedIndices(size_t batch_size, float alpha, float beta);
    float ComputeMaxWeight(float beta) const;
};

} // namespace utils
