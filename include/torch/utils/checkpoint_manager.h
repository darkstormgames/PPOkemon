#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace utils {

struct CheckpointData {
    std::string model_path;
    std::string optimizer_path;
    std::string metadata_path;
    int64_t step;
    float metric_value;
    std::chrono::system_clock::time_point timestamp;
    
    CheckpointData(const std::string& model, const std::string& opt, const std::string& meta,
                   int64_t s, float metric, std::chrono::system_clock::time_point ts)
        : model_path(model), optimizer_path(opt), metadata_path(meta), 
          step(s), metric_value(metric), timestamp(ts) {}
};

class CheckpointManager {
public:
    CheckpointManager(const std::string& checkpoint_dir, 
                     int max_to_keep = 5,
                     const std::string& metric_name = "reward");
    ~CheckpointManager() = default;
    
    // Save checkpoint
    std::string SaveCheckpoint(std::shared_ptr<torch::nn::Module> model, 
                              torch::optim::Optimizer& optimizer,
                              int64_t step,
                              float metric_value,
                              const std::unordered_map<std::string, float>& additional_metrics = {});
    
    // Load checkpoint
    bool LoadCheckpoint(const std::string& checkpoint_path,
                       std::shared_ptr<torch::nn::Module> model,
                       torch::optim::Optimizer& optimizer,
                       int64_t& step,
                       float& metric_value);
    
    // Load latest checkpoint
    bool LoadLatestCheckpoint(std::shared_ptr<torch::nn::Module> model,
                             torch::optim::Optimizer& optimizer,
                             int64_t& step,
                             float& metric_value);
    
    // Load best checkpoint (based on metric)
    bool LoadBestCheckpoint(std::shared_ptr<torch::nn::Module> model,
                           torch::optim::Optimizer& optimizer,
                           int64_t& step,
                           float& metric_value);
    
    // Checkpoint management
    std::vector<std::string> ListCheckpoints() const;
    void CleanupOldCheckpoints();
    void DeleteCheckpoint(const std::string& checkpoint_path);
    void DeleteAllCheckpoints();
    
    // Utility methods
    bool HasCheckpoints() const;
    std::string GetLatestCheckpointPath() const;
    std::string GetBestCheckpointPath() const;
    
    // Configuration
    void SetMaxToKeep(int max_to_keep) { max_to_keep_ = max_to_keep; }
    void SetMetricName(const std::string& metric_name) { metric_name_ = metric_name; }
    void SetSaveOnlyBest(bool save_only_best) { save_only_best_ = save_only_best; }
    
    // Statistics
    void PrintCheckpointStats() const;

private:
    std::string checkpoint_dir_;
    int max_to_keep_;
    std::string metric_name_;
    bool save_only_best_;
    bool higher_is_better_; // true for rewards, false for losses
    
    std::vector<CheckpointData> checkpoints_;
    
    // Helper methods
    void CreateCheckpointDirectory();
    std::string GenerateCheckpointPath(int64_t step) const;
    void LoadCheckpointRegistry();
    void SaveCheckpointRegistry();
    void SortCheckpointsByMetric();
    bool IsMetricBetter(float new_metric, float current_best) const;
    void SaveMetadata(const std::string& metadata_path, int64_t step, float metric_value,
                     const std::unordered_map<std::string, float>& additional_metrics);
    bool LoadMetadata(const std::string& metadata_path, int64_t& step, float& metric_value);
};

} // namespace utils
