#include "../../../include/torch/algorithms/rollout_buffer.h"
#include "../../../include/torch/utils/advantage_estimator.h"
#include <algorithm>
#include <random>
#include <iostream>

namespace algorithms {

RolloutBuffer::RolloutBuffer(const Config& config) 
    : config_(config), step_(0), finalized_(false) {
    InitializeBuffers();
}

void RolloutBuffer::InitializeBuffers() {
    // We'll initialize tensor storage when we get the first observation
    // This allows for flexible observation shapes
    step_ = 0;
    finalized_ = false;
}

void RolloutBuffer::Add(const torch::Tensor& observations,
                       const torch::Tensor& actions,
                       const torch::Tensor& action_log_probs,
                       const torch::Tensor& values,
                       const torch::Tensor& rewards,
                       const torch::Tensor& dones) {
    
    if (step_ >= config_.buffer_size) {
        throw std::runtime_error("RolloutBuffer is full! Call Reset() before adding more data.");
    }
    
    // Initialize buffers on first addition
    if (step_ == 0) {
        auto obs_shape = observations.sizes().vec();
        auto action_shape = actions.sizes().vec();
        
        // Create buffer shapes: [buffer_size, num_envs, ...]
        std::vector<int64_t> obs_buffer_shape = {config_.buffer_size, config_.num_envs};
        std::vector<int64_t> scalar_buffer_shape = {config_.buffer_size, config_.num_envs};
        std::vector<int64_t> action_buffer_shape = {config_.buffer_size, config_.num_envs};
        
        // Add observation dimensions (skip the first dimension which should be num_envs)
        if (obs_shape.size() > 1) {
            obs_buffer_shape.insert(obs_buffer_shape.end(), obs_shape.begin() + 1, obs_shape.end());
        }
        
        // Add action dimensions (skip the first dimension which should be num_envs)
        if (action_shape.size() > 1) {
            action_buffer_shape.insert(action_buffer_shape.end(), action_shape.begin() + 1, action_shape.end());
        }
        
        // Initialize tensors on the specified device
        observations_ = torch::zeros(obs_buffer_shape, torch::TensorOptions().device(config_.device).dtype(observations.dtype()));
        actions_ = torch::zeros(action_buffer_shape, torch::TensorOptions().device(config_.device).dtype(actions.dtype()));
        action_log_probs_ = torch::zeros(scalar_buffer_shape, torch::TensorOptions().device(config_.device));
        values_ = torch::zeros(scalar_buffer_shape, torch::TensorOptions().device(config_.device));
        rewards_ = torch::zeros(scalar_buffer_shape, torch::TensorOptions().device(config_.device));
        dones_ = torch::zeros(scalar_buffer_shape, torch::TensorOptions().device(config_.device));
    }
    
    // Store data at current step
    observations_[step_] = observations.to(config_.device);
    actions_[step_] = actions.to(config_.device);
    action_log_probs_[step_] = action_log_probs.to(config_.device);
    values_[step_] = values.to(config_.device);
    rewards_[step_] = rewards.to(config_.device);
    dones_[step_] = dones.to(config_.device);
    
    step_++;
}

void RolloutBuffer::FinishRollout(const torch::Tensor& last_values) {
    if (step_ == 0) {
        throw std::runtime_error("Cannot finalize empty rollout buffer!");
    }
    
    // Compute advantages using GAE
    ComputeGAE(last_values.to(config_.device));
    
    // Normalize advantages if requested
    if (config_.normalize_advantages) {
        NormalizeAdvantages();
    }
    
    finalized_ = true;
}

void RolloutBuffer::ComputeGAE(const torch::Tensor& last_values) {
    torch::NoGradGuard no_grad;
    
    // Initialize returns and advantages tensors
    auto buffer_shape = rewards_.sizes();
    returns_ = torch::zeros(buffer_shape, torch::TensorOptions().device(config_.device));
    advantages_ = torch::zeros(buffer_shape, torch::TensorOptions().device(config_.device));
    
    // Flatten tensors for easier computation: [buffer_size * num_envs]
    auto rewards_flat = rewards_.view({-1});
    auto values_flat = values_.view({-1});
    auto dones_flat = dones_.view({-1});
    auto returns_flat = returns_.view({-1});
    auto advantages_flat = advantages_.view({-1});
    
    // Process each environment separately
    for (int env_idx = 0; env_idx < config_.num_envs; ++env_idx) {
        // Extract sequences for this environment
        auto env_rewards = torch::zeros({step_}, torch::TensorOptions().device(config_.device));
        auto env_values = torch::zeros({step_ + 1}, torch::TensorOptions().device(config_.device));
        auto env_dones = torch::zeros({step_}, torch::TensorOptions().device(config_.device));
        
        // Fill environment-specific sequences
        for (int t = 0; t < step_; ++t) {
            int idx = t * config_.num_envs + env_idx;
            env_rewards[t] = rewards_flat[idx];
            env_values[t] = values_flat[idx];
            env_dones[t] = dones_flat[idx];
        }
        
        // Add last value
        env_values[step_] = last_values[env_idx];
        
        // Compute GAE for this environment
        auto env_advantages = utils::AdvantageEstimator::ComputeGAE(
            env_rewards, env_values.slice(0, 0, step_), env_dones, 
            config_.gamma, config_.gae_lambda
        );
        
        // Compute returns = advantages + values
        auto env_returns = env_advantages + env_values.slice(0, 0, step_);
        
        // Store back in flattened tensors
        for (int t = 0; t < step_; ++t) {
            int idx = t * config_.num_envs + env_idx;
            advantages_flat[idx] = env_advantages[t];
            returns_flat[idx] = env_returns[t];
        }
    }
    
    // Reshape back to original buffer shape
    advantages_ = advantages_flat.view(buffer_shape);
    returns_ = returns_flat.view(buffer_shape);
}

void RolloutBuffer::NormalizeAdvantages() {
    torch::NoGradGuard no_grad;
    
    if (advantages_.numel() == 0) {
        return;
    }
    
    // Compute mean and std across all steps and environments
    auto mean = advantages_.mean();
    auto std = advantages_.std();
    
    // Avoid division by zero
    std = torch::clamp_min(std, 1e-8);
    
    // Normalize
    advantages_ = (advantages_ - mean) / std;
}

std::vector<RolloutBuffer::Batch> RolloutBuffer::GetBatches(int batch_size, bool shuffle) {
    if (!finalized_) {
        throw std::runtime_error("RolloutBuffer must be finalized before getting batches!");
    }
    
    // Total number of samples
    int64_t total_samples = step_ * config_.num_envs;
    
    // Generate indices
    std::vector<int64_t> indices;
    if (shuffle) {
        indices = GetShuffledIndices(total_samples);
    } else {
        indices.resize(total_samples);
        std::iota(indices.begin(), indices.end(), 0);
    }
    
    std::vector<Batch> batches;
    
    // Create mini-batches
    for (int64_t start = 0; start < total_samples; start += batch_size) {
        int64_t end = std::min(start + batch_size, total_samples);
        int64_t actual_batch_size = end - start;
        
        if (actual_batch_size == 0) break;
        
        // Flatten all tensors for indexing: [buffer_size * num_envs, ...]
        auto obs_flat = observations_.view({total_samples, -1});
        auto actions_flat = actions_.view({total_samples, -1});
        auto log_probs_flat = action_log_probs_.view({total_samples});
        auto values_flat = values_.view({total_samples});
        auto returns_flat = returns_.view({total_samples});
        auto advantages_flat = advantages_.view({total_samples});
        
        // Create index tensor for this batch
        auto batch_indices = torch::tensor(
            std::vector<int64_t>(indices.begin() + start, indices.begin() + end),
            torch::TensorOptions().device(config_.device).dtype(torch::kLong)
        );
        
        // Extract batch data
        Batch batch;
        batch.observations = obs_flat.index_select(0, batch_indices);
        batch.actions = actions_flat.index_select(0, batch_indices);
        batch.old_action_log_probs = log_probs_flat.index_select(0, batch_indices);
        batch.old_values = values_flat.index_select(0, batch_indices);
        batch.returns = returns_flat.index_select(0, batch_indices);
        batch.advantages = advantages_flat.index_select(0, batch_indices);
        
        // Reshape observations and actions back to proper shape if needed
        if (observations_.dim() > 2) {
            auto obs_shape = observations_.sizes().vec();
            obs_shape[0] = actual_batch_size;
            obs_shape.erase(obs_shape.begin() + 1); // Remove the num_envs dimension
            batch.observations = batch.observations.view(obs_shape);
        }
        
        if (actions_.dim() > 2) {
            auto action_shape = actions_.sizes().vec();
            action_shape[0] = actual_batch_size;
            action_shape.erase(action_shape.begin() + 1); // Remove the num_envs dimension
            batch.actions = batch.actions.view(action_shape);
        }
        
        batches.push_back(std::move(batch));
    }
    
    return batches;
}

void RolloutBuffer::Reset() {
    step_ = 0;
    finalized_ = false;
    
    // Clear computed tensors
    returns_ = torch::Tensor();
    advantages_ = torch::Tensor();
}

RolloutBuffer::Stats RolloutBuffer::GetStats() const {
    Stats stats;
    
    if (step_ == 0) {
        return stats;
    }
    
    torch::NoGradGuard no_grad;
    
    // Compute basic statistics
    stats.mean_reward = rewards_.slice(0, 0, step_).mean().item<float>();
    stats.mean_value = values_.slice(0, 0, step_).mean().item<float>();
    
    if (finalized_ && advantages_.defined()) {
        stats.mean_advantage = advantages_.mean().item<float>();
        stats.std_advantage = advantages_.std().item<float>();
    }
    
    // Count completed episodes (where done = 1)
    auto dones_slice = dones_.slice(0, 0, step_);
    stats.num_episodes = dones_slice.sum().item<int>();
    
    return stats;
}

std::vector<int64_t> RolloutBuffer::GetShuffledIndices(int64_t size) const {
    std::vector<int64_t> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    
    // Use random device for shuffling
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    return indices;
}

} // namespace algorithms