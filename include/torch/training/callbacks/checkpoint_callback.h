#pragma once

#include <string>
#include <memory>
#include <functional>

namespace training {

/**
 * Base class for training callbacks
 * Callbacks are called at specific points during training
 */
class Callback {
public:
    virtual ~Callback() = default;
    
    // Called before training starts
    virtual void OnTrainBegin() {}
    
    // Called after training ends
    virtual void OnTrainEnd() {}
    
    // Called at the beginning of each step
    virtual void OnStepBegin(int64_t /*step*/) {}
    
    // Called at the end of each step
    virtual void OnStepEnd(int64_t /*step*/, float /*reward*/, float /*loss*/) {}
    
    // Called during evaluation
    virtual void OnEvaluation(int64_t /*step*/, float /*eval_reward*/) {}
};

/**
 * Checkpoint callback for saving models during training
 */
class CheckpointCallback : public Callback {
public:
    struct Config {
        std::string checkpoint_dir = "./checkpoints";
        std::string metric_name = "reward";
        int save_interval = 50000;  // Steps between saves
        int max_to_keep = 5;
        bool save_only_best = false;
        bool higher_is_better = true;
        float min_improvement = 0.01f;  // Minimum improvement to save
    };

    CheckpointCallback(const Config& config);
    ~CheckpointCallback() = default;

    // Callback interface
    void OnTrainBegin() override;
    void OnStepEnd(int64_t step, float reward, float loss) override;
    void OnEvaluation(int64_t step, float eval_reward) override;
    void OnTrainEnd() override;

    // Model saving interface
    using SaveModelFunction = std::function<bool(const std::string& path)>;
    void SetSaveModelFunction(const SaveModelFunction& save_fn) { save_model_fn_ = save_fn; }

    // Configuration
    void SetMetricName(const std::string& metric_name) { config_.metric_name = metric_name; }
    void SetSaveInterval(int interval) { config_.save_interval = interval; }
    void SetMaxToKeep(int max_to_keep) { config_.max_to_keep = max_to_keep; }

    // Statistics
    float GetBestMetric() const { return best_metric_; }
    int64_t GetLastSaveStep() const { return last_save_step_; }
    std::string GetBestCheckpointPath() const { return best_checkpoint_path_; }

private:
    Config config_;
    SaveModelFunction save_model_fn_;
    
    float best_metric_;
    int64_t last_save_step_;
    std::string best_checkpoint_path_;
    
    // Helper methods
    bool ShouldSave(int64_t step, float metric_value) const;
    void SaveCheckpoint(int64_t step, float metric_value);
    std::string GenerateCheckpointPath(int64_t step) const;
    void CleanupOldCheckpoints() const;
    void LogCheckpoint(const std::string& path, int64_t step, float metric) const;
};

} // namespace training
