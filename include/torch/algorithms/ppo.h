#pragma once

#include <torch/torch.h>
#include "../networks/network_base.h"
#include "../envs/env_abstract.h"
#include "rollout_buffer.h"
#include "../training/trainer.h"
#include <memory>
#include <vector>

namespace algorithms {

/**
 * Proximal Policy Optimization (PPO) Algorithm Implementation
 * Features:
 * - Clipped surrogate objective
 * - Adaptive KL penalty (optional)
 * - Value function clipping
 * - GPU acceleration support
 * - Multiple environments support
 * - Entropy regularization
 */
class PPO : public training::Trainer {
public:
    struct Config {
        // PPO-specific hyperparameters
        float clip_ratio = 0.2f;           // PPO clipping parameter
        float value_clip_ratio = 0.2f;     // Value function clipping parameter
        float entropy_coef = 0.01f;        // Entropy regularization coefficient
        float value_coef = 0.5f;           // Value loss coefficient
        float max_grad_norm = 0.5f;        // Gradient clipping norm
        
        // Training parameters
        int ppo_epochs = 4;                // Number of PPO update epochs per rollout
        int mini_batch_size = 64;          // Mini-batch size for PPO updates
        
        // Learning rate schedule
        float learning_rate = 3e-4f;       // Initial learning rate
        bool use_lr_schedule = true;       // Whether to use learning rate scheduling
        
        // KL divergence (optional)
        bool use_kl_penalty = false;       // Whether to use adaptive KL penalty
        float target_kl = 0.01f;           // Target KL divergence
        float kl_coef = 0.2f;             // KL penalty coefficient
        
        // Normalization
        bool normalize_observations = true; // Normalize observations
        bool normalize_rewards = false;     // Normalize rewards
        bool normalize_advantages = true;   // Normalize advantages
        
        // Device and environment
        torch::Device device = torch::kCPU;
        int num_envs = 8;                   // Number of parallel environments
        int rollout_steps = 2048;           // Steps per rollout
        
        // Evaluation
        int eval_frequency = 10;            // Evaluate every N updates
        int eval_episodes = 5;              // Number of episodes for evaluation
        
        // Logging and checkpointing
        bool verbose = true;
        int log_frequency = 10;             // Log every N updates
        int save_frequency = 50;            // Save model every N updates
        std::string checkpoint_dir = "./checkpoints";
        std::string log_dir = "./logs";
    };

    PPO(std::shared_ptr<networks::NetworkBase> policy_network,
        const Config& config);
    
    ~PPO() = default;

    // Trainer interface implementation
    void Train() override;
    void Evaluate(int num_episodes = 10) override;
    
    // PPO-specific training interface
    void TrainWithEnvironments(std::vector<std::unique_ptr<AbstractEnv>>& environments,
                              int total_updates = 1000);
    
    // Evaluate the current policy
    float EvaluateWithEnvironments(std::vector<std::unique_ptr<AbstractEnv>>& eval_environments);
    
    // Save/Load model
    void SaveModel(const std::string& path);
    void LoadModel(const std::string& path);
    
    // Configuration
    void SetDevice(const torch::Device& device);
    const Config& GetConfig() const { return config_; }
    
    // Statistics
    struct TrainingStats {
        float policy_loss = 0.0f;
        float value_loss = 0.0f;
        float entropy_loss = 0.0f;
        float total_loss = 0.0f;
        float clip_fraction = 0.0f;
        float kl_divergence = 0.0f;
        float explained_variance = 0.0f;
        float average_reward = 0.0f;
        float average_episode_length = 0.0f;
        int episodes_completed = 0;
    };
    
    const TrainingStats& GetLastStats() const { return last_stats_; }

protected:
    // Trainer interface implementation
    void CollectExperience() override;
    float UpdatePolicy() override;
    void ResetEnvironments() override;

private:
    Config config_;
    std::shared_ptr<networks::NetworkBase> policy_network_;
    std::unique_ptr<torch::optim::Adam> optimizer_;
    std::unique_ptr<RolloutBuffer> rollout_buffer_;
    
    // Environment management
    std::vector<std::unique_ptr<AbstractEnv>>* current_environments_;
    std::vector<torch::Tensor> current_observations_;
    std::vector<bool> environment_dones_;
    
    // Statistics tracking
    TrainingStats last_stats_;
    std::vector<float> episode_rewards_;
    std::vector<float> episode_lengths_;
    
    // Training state
    int update_count_;
    float current_lr_;
    
    // Helper methods
    void InitializeTraining();
    void InitializeEnvironments(std::vector<std::unique_ptr<AbstractEnv>>& environments);
    
    // Policy methods
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
    EvaluateActions(const torch::Tensor& observations, const torch::Tensor& actions);
    
    torch::Tensor GetActions(const torch::Tensor& observations);
    torch::Tensor GetValues(const torch::Tensor& observations);
    
    // Loss computation
    torch::Tensor ComputePolicyLoss(const RolloutBuffer::Batch& batch);
    torch::Tensor ComputeValueLoss(const RolloutBuffer::Batch& batch);
    torch::Tensor ComputeEntropyLoss(const RolloutBuffer::Batch& batch);
    
    // PPO-specific computations
    torch::Tensor ComputeClippedPolicyLoss(const torch::Tensor& log_probs,
                                          const torch::Tensor& old_log_probs,
                                          const torch::Tensor& advantages);
    
    torch::Tensor ComputeClippedValueLoss(const torch::Tensor& values,
                                         const torch::Tensor& old_values,
                                         const torch::Tensor& returns);
    
    float ComputeKLDivergence(const torch::Tensor& log_probs,
                             const torch::Tensor& old_log_probs);
    
    // Learning rate scheduling
    void UpdateLearningRate();
    
    // Statistics and logging
    void UpdateStatistics(const TrainingStats& stats);
    void LogTrainingProgress();
    
    // Action distribution utilities
    torch::Tensor ComputeLogProbs(const torch::Tensor& action_logits, 
                                 const torch::Tensor& actions);
    
    torch::Tensor ComputeEntropy(const torch::Tensor& action_logits);
    
    // Environment interaction
    void StepEnvironments();
    void CollectRollout();
    
    // GPU/CPU utilities
    void MoveToDevice(const torch::Device& device);
    torch::Tensor ToDevice(const torch::Tensor& tensor);
};

} // namespace algorithms
