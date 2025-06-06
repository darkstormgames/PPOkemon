#pragma once

#include "torch/envs/env_abstract.h"
#include <memory>
#include <vector>
#include <torch/torch.h>

/// @brief Struct to hold vectorized step results
struct VectorizedStepResult
{
    torch::Tensor rewards;      // [num_envs]
    torch::Tensor dones;        // [num_envs] (bool tensor)
    torch::Tensor tot_rewards;  // [num_envs] (only filled for done episodes)
    torch::Tensor tot_steps;    // [num_envs] (only filled for done episodes)
};

/// @brief Vectorized environment wrapper that runs multiple environments in parallel
class VectorizedEnv
{
public:
    /// @brief Constructor that creates multiple instances of an environment
    /// @param env_factory Function that creates a new environment instance
    /// @param num_envs Number of environments to run in parallel
    /// @param device Device to place tensors on (CPU or CUDA)
    VectorizedEnv(std::function<std::unique_ptr<AbstractEnv>()> env_factory, 
                  int num_envs, 
                  const torch::Device& device = torch::kCPU);
    
    ~VectorizedEnv();

    /// @brief Get the total number of environments
    int GetNumEnvs() const { return num_envs_; }
    
    /// @brief Get observation size for a single environment
    int64_t GetObservationSize() const;
    
    /// @brief Get observation shape for a single environment
    std::vector<int64_t> GetObservationShape() const;
    
    /// @brief Get action size for a single environment
    int64_t GetActionSize() const;
    
    /// @brief Reset all environments and get initial observations
    /// @return Tensor of shape [num_envs, obs_size] or [num_envs, ...obs_shape]
    torch::Tensor Reset();
    
    /// @brief Step all environments with given actions
    /// @param actions Tensor of shape [num_envs, action_size]
    /// @return Tuple of (observations, step_results)
    std::tuple<torch::Tensor, VectorizedStepResult> Step(const torch::Tensor& actions);
    
    /// @brief Render all environments
    /// @param wait_ms Milliseconds to wait after rendering
    void Render(uint64_t wait_ms = 0);
    
    /// @brief Set seed for all environments
    /// @param seed Base seed (each env gets seed + env_index)
    void SetSeed(unsigned int seed);

private:
    std::vector<std::unique_ptr<AbstractEnv>> envs_;
    int num_envs_;
    torch::Device device_;
    
    // Cache for observation and action buffers
    std::vector<std::vector<float>> obs_buffers_;
    std::vector<std::vector<float>> action_buffers_;
};
