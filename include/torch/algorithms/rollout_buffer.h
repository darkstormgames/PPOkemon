#pragma once

#include <torch/torch.h>
#include <vector>
#include <memory>

namespace algorithms {

/**
 * RolloutBuffer for storing and processing trajectories during PPO training
 * Supports GAE (Generalized Advantage Estimation) computation and GPU acceleration
 */
class RolloutBuffer {
public:
    struct Config {
        int buffer_size = 2048;      // Number of steps to collect before update
        int num_envs = 8;            // Number of parallel environments
        float gamma = 0.99f;         // Discount factor
        float gae_lambda = 0.95f;    // GAE lambda parameter
        bool normalize_advantages = true;  // Whether to normalize advantages
        torch::Device device = torch::kCPU;  // Device for computations
    };

    explicit RolloutBuffer(const Config& config);
    ~RolloutBuffer() = default;

    // Add a single step to the buffer
    void Add(const torch::Tensor& observations,
             const torch::Tensor& actions,
             const torch::Tensor& action_log_probs,
             const torch::Tensor& values,
             const torch::Tensor& rewards,
             const torch::Tensor& dones);

    // Finalize the buffer and compute advantages
    void FinishRollout(const torch::Tensor& last_values);

    // Get mini-batches for training
    struct Batch {
        torch::Tensor observations;
        torch::Tensor actions;
        torch::Tensor old_action_log_probs;
        torch::Tensor old_values;
        torch::Tensor returns;
        torch::Tensor advantages;
        
        // Move batch to device
        void to(const torch::Device& device) {
            observations = observations.to(device);
            actions = actions.to(device);
            old_action_log_probs = old_action_log_probs.to(device);
            old_values = old_values.to(device);
            returns = returns.to(device);
            advantages = advantages.to(device);
        }
    };

    // Get iterator for mini-batches
    std::vector<Batch> GetBatches(int batch_size, bool shuffle = true);
    
    // Clear the buffer
    void Reset();
    
    // Utility functions
    int Size() const { return step_; }
    bool IsFull() const { return step_ >= config_.buffer_size; }
    torch::Device GetDevice() const { return config_.device; }
    
    // Get statistics
    struct Stats {
        float mean_reward = 0.0f;
        float mean_value = 0.0f;
        float mean_advantage = 0.0f;
        float std_advantage = 1.0f;
        int num_episodes = 0;
    };
    Stats GetStats() const;

private:
    Config config_;
    int step_;
    
    // Storage tensors (buffer_size x num_envs x ...)
    torch::Tensor observations_;
    torch::Tensor actions_;
    torch::Tensor action_log_probs_;
    torch::Tensor values_;
    torch::Tensor rewards_;
    torch::Tensor dones_;
    
    // Computed during finalization
    torch::Tensor returns_;
    torch::Tensor advantages_;
    bool finalized_;
    
    // Helper methods
    void InitializeBuffers();
    void ComputeGAE(const torch::Tensor& last_values);
    void NormalizeAdvantages();
    std::vector<int64_t> GetShuffledIndices(int64_t size) const;
};

} // namespace algorithms
