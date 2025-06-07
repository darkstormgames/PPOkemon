#include "../../../include/torch/algorithms/ppo.h"
#include "../../../include/torch/utils/advantage_estimator.h"
#include "../../../include/torch/networks/a2c.h"
#include <iostream>
#include <algorithm>
#include <numeric>

namespace algorithms {

PPO::PPO(std::shared_ptr<networks::NetworkBase> policy_network, const Config& config)
    : training::Trainer({
        .total_steps = config.rollout_steps * 1000, // Convert to total steps
        .batch_size = config.mini_batch_size,
        .learning_rate = config.learning_rate,
        .num_envs = config.num_envs,
        .device = config.device,
        .experiment_name = "ppo_training",
        .log_dir = config.log_dir,
        .checkpoint_dir = config.checkpoint_dir,
        .verbose = config.verbose
      }), 
      config_(config), policy_network_(policy_network), 
      current_environments_(nullptr), update_count_(0), current_lr_(config.learning_rate) {
    
    // Initialize optimizer
    optimizer_ = std::make_unique<torch::optim::Adam>(
        policy_network_->parameters(), 
        torch::optim::AdamOptions(config_.learning_rate)
    );
    
    // Initialize rollout buffer
    RolloutBuffer::Config buffer_config;
    buffer_config.buffer_size = config_.rollout_steps;
    buffer_config.num_envs = config_.num_envs;
    buffer_config.device = config_.device;
    buffer_config.normalize_advantages = config_.normalize_advantages;
    
    rollout_buffer_ = std::make_unique<RolloutBuffer>(buffer_config);
    
    // Move network to device
    policy_network_->to(config_.device);
    
    InitializeTraining();
}

void PPO::InitializeTraining() {
    // Reset training state
    update_count_ = 0;
    current_lr_ = config_.learning_rate;
    
    // Clear statistics
    last_stats_ = TrainingStats{};
    episode_rewards_.clear();
    episode_lengths_.clear();
    
    std::cout << "PPO Training initialized with:" << std::endl;
    std::cout << "  Device: " << config_.device << std::endl;
    std::cout << "  Environments: " << config_.num_envs << std::endl;
    std::cout << "  Rollout steps: " << config_.rollout_steps << std::endl;
    std::cout << "  Mini-batch size: " << config_.mini_batch_size << std::endl;
    std::cout << "  PPO epochs: " << config_.ppo_epochs << std::endl;
}

void PPO::TrainWithEnvironments(std::vector<std::unique_ptr<AbstractEnv>>& environments, int total_updates) {
    current_environments_ = &environments;
    InitializeEnvironments(environments);
    
    std::cout << "Starting PPO training for " << total_updates << " updates..." << std::endl;
    
    for (int update = 0; update < total_updates; ++update) {
        // Collect rollout
        CollectExperience();
        
        // Update policy
        float total_loss = UpdatePolicy();
        
        // Update learning rate
        if (config_.use_lr_schedule) {
            UpdateLearningRate();
        }
        
        // Log progress
        if (config_.verbose && (update + 1) % config_.log_frequency == 0) {
            LogTrainingProgress();
        }
        
        // Evaluate policy
        if ((update + 1) % config_.eval_frequency == 0) {
            float eval_reward = EvaluateWithEnvironments(environments);
            std::cout << "Update " << update + 1 << " - Evaluation reward: " << eval_reward << std::endl;
        }
        
        // Save checkpoint
        if ((update + 1) % config_.save_frequency == 0) {
            std::string checkpoint_path = config_.checkpoint_dir + "/ppo_update_" + std::to_string(update + 1) + ".pt";
            SaveModel(checkpoint_path);
        }
        
        update_count_++;
    }
    
    std::cout << "PPO training completed!" << std::endl;
}

// Trainer interface implementation
void PPO::Train() {
    // Default implementation - throw error if no environments are set
    if (!current_environments_ || current_environments_->empty()) {
        throw std::runtime_error("No environments set for training! Use TrainWithEnvironments() or set environments first.");
    }
    
    // Use default training parameters
    TrainWithEnvironments(*current_environments_, 1000);
}

void PPO::Evaluate(int num_episodes) {
    // Default implementation - throw error if no environments are set
    if (!current_environments_ || current_environments_->empty()) {
        throw std::runtime_error("No environments set for evaluation! Use EvaluateWithEnvironments() or set environments first.");
    }
    
    // Temporarily change eval episodes config
    int original_eval_episodes = config_.eval_episodes;
    config_.eval_episodes = num_episodes;
    
    float avg_reward = EvaluateWithEnvironments(*current_environments_);
    
    // Restore original config
    config_.eval_episodes = original_eval_episodes;
    
    std::cout << "Evaluation completed with average reward: " << avg_reward << std::endl;
}

void PPO::InitializeEnvironments(std::vector<std::unique_ptr<AbstractEnv>>& environments) {
    if (static_cast<int>(environments.size()) != config_.num_envs) {
        throw std::runtime_error("Number of environments doesn't match config!");
    }
    
    current_observations_.clear();
    environment_dones_.clear();
    
    // Reset all environments and collect initial observations
    for (auto& env : environments) {
        env->Reset();
        
        // Get observation data from environment
        std::vector<float> obs_data(env->GetObservationSize());
        env->GetObsData(obs_data.data());
        
        // Convert to tensor
        auto obs_tensor = torch::from_blob(obs_data.data(), {env->GetObservationSize()}, torch::kFloat32).clone();
        current_observations_.push_back(obs_tensor.to(config_.device));
        environment_dones_.push_back(false);
    }
}

void PPO::CollectExperience() {
    rollout_buffer_->Reset();
    
    // Collect rollout_steps worth of data
    for (int step = 0; step < config_.rollout_steps; ++step) {
        // Stack observations for batch processing
        auto batch_obs = torch::stack(current_observations_);
        
        // Get actions and values from policy
        torch::NoGradGuard no_grad;
        auto actions = GetActions(batch_obs);
        auto values = GetValues(batch_obs);
        auto action_log_probs = ComputeLogProbs(policy_network_->forward(batch_obs), actions);
        
        // Step environments
        std::vector<torch::Tensor> next_observations;
        next_observations.reserve(config_.num_envs); // Reserve space for efficiency
        std::vector<float> rewards;
        rewards.reserve(config_.num_envs);
        std::vector<bool> dones;
        dones.reserve(config_.num_envs);
        
        for (int env_idx = 0; env_idx < config_.num_envs; ++env_idx) {
            if (!environment_dones_[env_idx]) {
                // Convert action tensor to action data for environment
                auto action_tensor = actions[env_idx].to(torch::kCPU);
                
                // Convert action tensor to float for environment interface
                // Actions might be Long (int64) from multinomial sampling
                auto action_float = action_tensor.to(torch::kFloat32);
                std::vector<float> action_data(action_float.numel());
                std::memcpy(action_data.data(), action_float.data_ptr<float>(), action_float.numel() * sizeof(float));
                
                // Step the environment
                auto step_result = (*current_environments_)[env_idx]->Step(action_data.data(), action_data.size());
                
                // Get next observation
                std::vector<float> next_obs_data((*current_environments_)[env_idx]->GetObservationSize());
                (*current_environments_)[env_idx]->GetObsData(next_obs_data.data());
                auto next_obs_tensor = torch::from_blob(next_obs_data.data(), 
                    {(*current_environments_)[env_idx]->GetObservationSize()}, torch::kFloat32).clone();
                
                next_observations.push_back(next_obs_tensor.to(config_.device));
                rewards.push_back(step_result.reward);
                dones.push_back(step_result.done);
                
                if (step_result.done) {
                    environment_dones_[env_idx] = true;
                    episode_rewards_.push_back(step_result.tot_reward);
                    episode_lengths_.push_back(static_cast<float>(step_result.tot_steps));
                    
                    // Reset environment
                    (*current_environments_)[env_idx]->Reset();
                    
                    // Get reset observation and update the already-pushed tensor
                    std::vector<float> reset_obs_data((*current_environments_)[env_idx]->GetObservationSize());
                    (*current_environments_)[env_idx]->GetObsData(reset_obs_data.data());
                    auto reset_obs_tensor = torch::from_blob(reset_obs_data.data(), 
                        {(*current_environments_)[env_idx]->GetObservationSize()}, torch::kFloat32).clone();
                    
                    // Update the observation that was already pushed
                    next_observations[env_idx] = reset_obs_tensor.to(config_.device);
                    environment_dones_[env_idx] = false;
                }
            } else {
                // Environment was done, use current observation
                next_observations.push_back(current_observations_[env_idx]);
                rewards.push_back(0.0f);
                dones.push_back(false);
            }
        }
        
        // Convert to tensors
        auto rewards_tensor = torch::tensor(rewards, torch::TensorOptions().device(config_.device));
        
        // Convert dones properly from vector<bool>
        std::vector<float> dones_float;
        dones_float.reserve(dones.size());
        for (bool done : dones) {
            dones_float.push_back(done ? 1.0f : 0.0f);
        }
        auto dones_tensor = torch::tensor(dones_float, torch::TensorOptions().device(config_.device));
        
        // Add to rollout buffer
        rollout_buffer_->Add(batch_obs, actions, action_log_probs, 
                           values.squeeze(-1), rewards_tensor, dones_tensor);
        
        // Update current observations
        current_observations_ = next_observations;
    }
    
    // Compute final values for GAE
    auto final_batch_obs = torch::stack(current_observations_);
    torch::NoGradGuard no_grad;
    auto final_values = GetValues(final_batch_obs).squeeze(-1);
    
    // Finalize rollout buffer
    rollout_buffer_->FinishRollout(final_values);
}

float PPO::UpdatePolicy() {
    float total_loss = 0.0f;
    TrainingStats epoch_stats;
    
    // Get all batches for this update
    auto batches = rollout_buffer_->GetBatches(config_.mini_batch_size, true);
    
    for (int epoch = 0; epoch < config_.ppo_epochs; ++epoch) {
        float epoch_policy_loss = 0.0f;
        float epoch_value_loss = 0.0f;
        float epoch_entropy_loss = 0.0f;
        float epoch_kl_div = 0.0f;
        int num_batches = 0;
        
        for (auto& batch : batches) {
            // Move batch to device
            batch.to(config_.device);
            
            // Zero gradients
            optimizer_->zero_grad();
            
            // Compute losses
            auto policy_loss = ComputePolicyLoss(batch);
            auto value_loss = ComputeValueLoss(batch);
            auto entropy_loss = ComputeEntropyLoss(batch);
            
            // Compute total loss
            auto loss = policy_loss + config_.value_coef * value_loss - config_.entropy_coef * entropy_loss;
            
            // Add KL penalty if enabled
            if (config_.use_kl_penalty) {
                auto kl_div = ComputeKLDivergence(
                    ComputeLogProbs(policy_network_->forward(batch.observations), batch.actions),
                    batch.old_action_log_probs
                );
                loss = loss + config_.kl_coef * kl_div;
                epoch_kl_div += kl_div;
            }
            
            // Backward pass
            loss.backward();
            
            // Gradient clipping
            if (config_.max_grad_norm > 0) {
                torch::nn::utils::clip_grad_norm_(policy_network_->parameters(), config_.max_grad_norm);
            }
            
            // Optimizer step
            optimizer_->step();
            
            // Accumulate statistics
            epoch_policy_loss += policy_loss.item<float>();
            epoch_value_loss += value_loss.item<float>();
            epoch_entropy_loss += entropy_loss.item<float>();
            total_loss += loss.item<float>();
            num_batches++;
        }
        
        // Early stopping on KL divergence
        if (config_.use_kl_penalty && epoch_kl_div / num_batches > 1.5 * config_.target_kl) {
            std::cout << "Early stopping at epoch " << epoch << " due to KL divergence" << std::endl;
            break;
        }
    }
    
    // Update statistics
    auto buffer_stats = rollout_buffer_->GetStats();
    last_stats_.policy_loss = total_loss / (config_.ppo_epochs * batches.size());
    last_stats_.value_loss = epoch_stats.value_loss;
    last_stats_.entropy_loss = epoch_stats.entropy_loss;
    last_stats_.total_loss = total_loss / (config_.ppo_epochs * batches.size());
    last_stats_.average_reward = buffer_stats.mean_reward;
    last_stats_.episodes_completed = buffer_stats.num_episodes;
    
    return total_loss / (config_.ppo_epochs * batches.size());
}

torch::Tensor PPO::ComputePolicyLoss(const RolloutBuffer::Batch& batch) {
    // Re-evaluate actions with current policy
    auto action_logits = policy_network_->forward(batch.observations);
    auto log_probs = ComputeLogProbs(action_logits, batch.actions);
    
    return ComputeClippedPolicyLoss(log_probs, batch.old_action_log_probs, batch.advantages);
}

torch::Tensor PPO::ComputeValueLoss(const RolloutBuffer::Batch& batch) {
    auto values = GetValues(batch.observations).squeeze(-1);
    
    if (config_.value_clip_ratio > 0) {
        return ComputeClippedValueLoss(values, batch.old_values, batch.returns);
    } else {
        // Standard MSE loss
        return torch::mse_loss(values, batch.returns);
    }
}

torch::Tensor PPO::ComputeEntropyLoss(const RolloutBuffer::Batch& batch) {
    auto action_logits = policy_network_->forward(batch.observations);
    return ComputeEntropy(action_logits);
}

torch::Tensor PPO::ComputeClippedPolicyLoss(const torch::Tensor& log_probs,
                                           const torch::Tensor& old_log_probs,
                                           const torch::Tensor& advantages) {
    // Compute probability ratios
    auto ratio = torch::exp(log_probs - old_log_probs);
    
    // Clipped surrogate loss
    auto surr1 = ratio * advantages;
    auto surr2 = torch::clamp(ratio, 1.0f - config_.clip_ratio, 1.0f + config_.clip_ratio) * advantages;
    
    return -torch::mean(torch::min(surr1, surr2));
}

torch::Tensor PPO::ComputeClippedValueLoss(const torch::Tensor& values,
                                          const torch::Tensor& old_values,
                                          const torch::Tensor& returns) {
    // Clipped value loss similar to policy clipping
    auto value_clipped = old_values + torch::clamp(
        values - old_values, -config_.value_clip_ratio, config_.value_clip_ratio
    );
    
    auto loss_unclipped = torch::pow(values - returns, 2);
    auto loss_clipped = torch::pow(value_clipped - returns, 2);
    
    return 0.5f * torch::mean(torch::max(loss_unclipped, loss_clipped));
}

float PPO::ComputeKLDivergence(const torch::Tensor& log_probs,
                              const torch::Tensor& old_log_probs) {
    auto kl_div = old_log_probs - log_probs;
    return torch::mean(kl_div).item<float>();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
PPO::EvaluateActions(const torch::Tensor& observations, const torch::Tensor& actions) {
    auto action_logits = policy_network_->forward(observations);
    auto log_probs = ComputeLogProbs(action_logits, actions);
    auto entropy = ComputeEntropy(action_logits);
    auto values = GetValues(observations);
    
    return std::make_tuple(log_probs, entropy, values);
}

torch::Tensor PPO::GetActions(const torch::Tensor& observations) {
    auto action_logits = policy_network_->forward(observations);
    
    // For discrete actions, sample from categorical distribution
    auto probs = torch::softmax(action_logits, -1);
    auto actions = torch::multinomial(probs, 1).squeeze(-1);
    
    return actions;
}

torch::Tensor PPO::GetValues(const torch::Tensor& observations) {
    // Assume the network has a separate value head or can return values
    // This would depend on the specific network architecture (e.g., A2C-style network)
    
    // For now, assume we have an A2C network that can provide values
    if (auto a2c_network = std::dynamic_pointer_cast<networks::A2CImpl>(policy_network_)) {
        return a2c_network->GetValue(observations);
    } else {
        // Fallback: use a simple value estimation (this should be replaced with proper value network)
        return torch::zeros({observations.size(0), 1}, torch::TensorOptions().device(config_.device));
    }
}

torch::Tensor PPO::ComputeLogProbs(const torch::Tensor& action_logits, 
                                   const torch::Tensor& actions) {
    auto log_probs = torch::log_softmax(action_logits, -1);
    
    // Ensure actions tensor has correct shape for gather operation
    // actions should be [batch_size] -> [batch_size, 1] for gathering along dim 1
    torch::Tensor actions_for_gather;
    if (actions.dim() == 1) {
        actions_for_gather = actions.unsqueeze(1);
    } else if (actions.dim() == 2 && actions.size(1) == 1) {
        actions_for_gather = actions;
    } else {
        throw std::runtime_error("Unsupported actions tensor shape in ComputeLogProbs");
    }
    
    // Gather log probabilities for the taken actions
    return log_probs.gather(1, actions_for_gather).squeeze(1);
}

torch::Tensor PPO::ComputeEntropy(const torch::Tensor& action_logits) {
    auto probs = torch::softmax(action_logits, -1);
    auto log_probs = torch::log_softmax(action_logits, -1);
    return -torch::mean(torch::sum(probs * log_probs, -1));
}

void PPO::UpdateLearningRate() {
    // Simple linear decay
    float decay_factor = 1.0f - static_cast<float>(update_count_) / 1000.0f; // Adjust based on total updates
    decay_factor = std::max(decay_factor, 0.1f); // Minimum 10% of original LR
    
    current_lr_ = config_.learning_rate * decay_factor;
    
    for (auto& param_group : optimizer_->param_groups()) {
        param_group.options().set_lr(current_lr_);
    }
}

float PPO::EvaluateWithEnvironments(std::vector<std::unique_ptr<AbstractEnv>>& eval_environments) {
    torch::NoGradGuard no_grad;
    policy_network_->eval();
    
    std::vector<float> episode_rewards;
    
    for (int episode = 0; episode < config_.eval_episodes; ++episode) {
        auto& env = eval_environments[episode % eval_environments.size()];
        
        // Reset environment
        env->Reset();
        
        // Get initial observation
        std::vector<float> obs_data(env->GetObservationSize());
        env->GetObsData(obs_data.data());
        auto obs = torch::from_blob(obs_data.data(), {env->GetObservationSize()}, torch::kFloat32).clone();
        
        float total_reward = 0.0f;
        bool done = false;
        int max_steps = 1000; // Prevent infinite loops
        int step_count = 0;
        
        while (!done && step_count < max_steps) {
            auto action_logits = policy_network_->forward(obs.unsqueeze(0).to(config_.device));
            auto action = torch::argmax(action_logits, -1); // Deterministic for evaluation
            
            // Convert action tensor to action data
            auto action_cpu = action.to(torch::kCPU);
            
            // Convert action tensor to float for environment interface
            // Actions from argmax are Long (int64), need to convert to float
            auto action_float = action_cpu.to(torch::kFloat32);
            std::vector<float> action_data(action_float.numel());
            std::memcpy(action_data.data(), action_float.data_ptr<float>(), action_float.numel() * sizeof(float));
            
            // Step environment
            auto step_result = env->Step(action_data.data(), action_data.size());
            
            // Get next observation
            env->GetObsData(obs_data.data());
            obs = torch::from_blob(obs_data.data(), {env->GetObservationSize()}, torch::kFloat32).clone();
            
            total_reward += step_result.reward;
            done = step_result.done;
            step_count++;
        }
        
        episode_rewards.push_back(total_reward);
    }
    
    policy_network_->train();
    
    return std::accumulate(episode_rewards.begin(), episode_rewards.end(), 0.0f) / episode_rewards.size();
}

void PPO::SaveModel(const std::string& path) {
    torch::save(policy_network_, path);
    std::cout << "Model saved to: " << path << std::endl;
}

void PPO::LoadModel(const std::string& path) {
    torch::load(policy_network_, path);
    policy_network_->to(config_.device);
    std::cout << "Model loaded from: " << path << std::endl;
}

void PPO::SetDevice(const torch::Device& device) {
    config_.device = device;
    policy_network_->to(device);
    rollout_buffer_ = std::make_unique<RolloutBuffer>(RolloutBuffer::Config{
        .buffer_size = config_.rollout_steps,
        .num_envs = config_.num_envs,
        .gamma = 0.99f,
        .gae_lambda = 0.95f,
        .normalize_advantages = config_.normalize_advantages,
        .device = device
    });
}

void PPO::LogTrainingProgress() {
    std::cout << "Update " << update_count_ 
              << " | Loss: " << last_stats_.total_loss
              << " | Reward: " << last_stats_.average_reward
              << " | Episodes: " << last_stats_.episodes_completed
              << " | LR: " << current_lr_ << std::endl;
}

void PPO::ResetEnvironments() {
    if (current_environments_) {
        InitializeEnvironments(*current_environments_);
    }
}

torch::Tensor PPO::ToDevice(const torch::Tensor& tensor) {
    return tensor.to(config_.device);
}

void PPO::MoveToDevice(const torch::Device& device) {
    SetDevice(device);
}

} // namespace algorithms