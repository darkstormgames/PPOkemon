#pragma once

#include <torch/torch.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>

// Forward declarations
namespace utils {
    class TrainingLogger;
    class CheckpointManager;
    class LearningRateScheduler;
    class Profiler;
}

namespace networks {
    class NetworkBase;
}

class AbstractEnv;

namespace training {

/**
 * Base trainer class for reinforcement learning algorithms
 * Provides common functionality for training loops, logging, checkpointing, etc.
 */
class Trainer {
public:
    struct Config {
        // Training parameters
        int64_t total_steps = 1000000;
        int64_t batch_size = 32;
        int64_t eval_interval = 10000;
        int64_t save_interval = 50000;
        int64_t log_interval = 1000;
        
        // Optimization
        float learning_rate = 3e-4f;
        float weight_decay = 0.0f;
        float gradient_clip_norm = 0.5f;
        
        // Environment
        int num_envs = 8;
        int max_episode_steps = 1000;
        
        // Device
        torch::Device device = torch::kCPU;
        
        // Logging
        std::string experiment_name = "rl_training";
        std::string log_dir = "./logs";
        std::string checkpoint_dir = "./checkpoints";
        bool verbose = true;
        
        // Early stopping
        bool use_early_stopping = false;
        float early_stopping_patience = 100000;
        std::string early_stopping_metric = "reward";
    };

    Trainer(const Config& config);
    virtual ~Trainer();

    // Main training interface
    virtual void Train() = 0;
    virtual void Evaluate(int num_episodes = 10) = 0;
    
    // Training control
    void SetTotalSteps(int64_t steps) { config_.total_steps = steps; }
    void SetDevice(const torch::Device& device) { config_.device = device; }
    void Stop() { should_stop_ = true; }
    bool ShouldStop() const { return should_stop_; }
    
    // Model management
    virtual void SaveModel(const std::string& path) = 0;
    virtual void LoadModel(const std::string& path) = 0;
    
    // Statistics
    float GetCurrentReward() const { return current_reward_; }
    int64_t GetCurrentStep() const { return current_step_; }
    float GetLearningRate() const;
    
    // Callbacks
    using StepCallback = std::function<void(int64_t step, float reward, float loss)>;
    using EvalCallback = std::function<void(int64_t step, float eval_reward)>;
    
    void RegisterStepCallback(const StepCallback& callback) { step_callbacks_.push_back(callback); }
    void RegisterEvalCallback(const EvalCallback& callback) { eval_callbacks_.push_back(callback); }

protected:
    // Core training methods (to be implemented by subclasses)
    virtual void CollectExperience() = 0;
    virtual float UpdatePolicy() = 0;
    virtual void ResetEnvironments() = 0;
    
    // Utility methods
    void InitializeTraining();
    void LogMetrics(const std::unordered_map<std::string, float>& metrics);
    void CheckpointModel(float metric_value);
    void UpdateLearningRate();
    bool CheckEarlyStopping();
    
    // Member variables
    Config config_;
    int64_t current_step_;
    float current_reward_;
    float best_reward_;
    bool should_stop_;
    
    // Components
    std::unique_ptr<utils::TrainingLogger> logger_;
    std::unique_ptr<utils::CheckpointManager> checkpoint_manager_;
    std::unique_ptr<utils::LearningRateScheduler> lr_scheduler_;
    
    // Callbacks
    std::vector<StepCallback> step_callbacks_;
    std::vector<EvalCallback> eval_callbacks_;
    
    // Early stopping
    int64_t steps_without_improvement_;
    float early_stopping_best_metric_;
};

} // namespace training
