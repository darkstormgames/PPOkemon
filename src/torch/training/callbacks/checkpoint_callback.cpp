// filepath: /home/timo/PPOkemon/src/torch/training/callbacks/checkpoint_callback.cpp

#include "torch/training/callbacks/checkpoint_callback.h"
#include <filesystem>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace training {

CheckpointCallback::CheckpointCallback(const Config& config) 
    : config_(config), best_metric_(config.higher_is_better ? -std::numeric_limits<float>::infinity() 
                                                             : std::numeric_limits<float>::infinity()),
      last_save_step_(0) {
    // Create checkpoint directory
    std::filesystem::create_directories(config_.checkpoint_dir);
}

// ============================================================================
// Callback Interface Implementation
// ============================================================================

void CheckpointCallback::OnTrainBegin() {
    best_metric_ = config_.higher_is_better ? -std::numeric_limits<float>::infinity() 
                                            : std::numeric_limits<float>::infinity();
    last_save_step_ = 0;
    best_checkpoint_path_.clear();
    
    std::cout << "CheckpointCallback initialized:\n";
    std::cout << "  Directory: " << config_.checkpoint_dir << "\n";
    std::cout << "  Metric: " << config_.metric_name << " (higher_is_better: " 
              << (config_.higher_is_better ? "true" : "false") << ")\n";
    std::cout << "  Save interval: " << config_.save_interval << " steps\n";
    std::cout << "  Max to keep: " << config_.max_to_keep << "\n";
    std::cout << "  Save only best: " << (config_.save_only_best ? "true" : "false") << "\n\n";
}

void CheckpointCallback::OnStepEnd(int64_t step, float reward, float loss) {
    // Use reward as the default metric if metric_name is "reward"
    if (config_.metric_name == "reward") {
        if (ShouldSave(step, reward)) {
            SaveCheckpoint(step, reward);
        }
    } else if (config_.metric_name == "loss") {
        // For loss, lower is better (invert the logic)
        float inverted_loss = -loss;
        if (ShouldSave(step, inverted_loss)) {
            SaveCheckpoint(step, loss);
        }
    }
}

void CheckpointCallback::OnEvaluation(int64_t step, float eval_reward) {
    // Use evaluation reward for checkpointing
    if (config_.metric_name == "eval_reward" || config_.metric_name == "reward") {
        if (ShouldSave(step, eval_reward)) {
            SaveCheckpoint(step, eval_reward);
        }
    }
}

void CheckpointCallback::OnTrainEnd() {
    std::cout << "\nCheckpointCallback summary:\n";
    std::cout << "  Best " << config_.metric_name << ": " << best_metric_ << "\n";
    std::cout << "  Best checkpoint: " << best_checkpoint_path_ << "\n";
    std::cout << "  Last save step: " << last_save_step_ << "\n\n";
}

// ============================================================================
// Checkpoint Management
// ============================================================================

bool CheckpointCallback::ShouldSave(int64_t step, float metric_value) const {
    // Check if enough steps have passed since last save
    if (step - last_save_step_ < config_.save_interval) {
        return false;
    }
    
    // If save_only_best is true, only save if this is the best metric so far
    if (config_.save_only_best) {
        bool is_better;
        if (config_.higher_is_better) {
            is_better = metric_value > best_metric_ + config_.min_improvement;
        } else {
            is_better = metric_value < best_metric_ - config_.min_improvement;
        }
        return is_better;
    }
    
    // Otherwise, save at regular intervals
    return true;
}

void CheckpointCallback::SaveCheckpoint(int64_t step, float metric_value) {
    if (!save_model_fn_) {
        std::cerr << "Warning: SaveModelFunction not set, cannot save checkpoint\n";
        return;
    }
    
    std::string checkpoint_path = GenerateCheckpointPath(step);
    
    try {
        bool success = save_model_fn_(checkpoint_path);
        
        if (success) {
            last_save_step_ = step;
            
            // Update best metric and path if this is better
            bool is_better;
            if (config_.higher_is_better) {
                is_better = metric_value > best_metric_;
            } else {
                is_better = metric_value < best_metric_;
            }
            
            if (is_better) {
                best_metric_ = metric_value;
                best_checkpoint_path_ = checkpoint_path;
            }
            
            LogCheckpoint(checkpoint_path, step, metric_value);
            CleanupOldCheckpoints();
            
        } else {
            std::cerr << "Failed to save checkpoint at step " << step << "\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error saving checkpoint at step " << step << ": " << e.what() << "\n";
    }
}

std::string CheckpointCallback::GenerateCheckpointPath(int64_t step) const {
    std::ostringstream oss;
    oss << config_.checkpoint_dir << "/checkpoint_" << std::setfill('0') << std::setw(8) << step;
    return oss.str();
}

void CheckpointCallback::CleanupOldCheckpoints() const {
    if (config_.max_to_keep <= 0) return;
    
    try {
        std::vector<std::filesystem::path> checkpoint_files;
        
        // Collect all checkpoint files
        for (const auto& entry : std::filesystem::directory_iterator(config_.checkpoint_dir)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if (filename.find("checkpoint_") == 0) {
                    checkpoint_files.push_back(entry.path());
                }
            }
        }
        
        // Sort by modification time (oldest first)
        std::sort(checkpoint_files.begin(), checkpoint_files.end(),
                 [](const std::filesystem::path& a, const std::filesystem::path& b) {
                     return std::filesystem::last_write_time(a) < std::filesystem::last_write_time(b);
                 });
        
        // Remove excess files
        if (checkpoint_files.size() > static_cast<size_t>(config_.max_to_keep)) {
            size_t num_to_remove = checkpoint_files.size() - config_.max_to_keep;
            for (size_t i = 0; i < num_to_remove; ++i) {
                // Don't remove the best checkpoint
                if (checkpoint_files[i].string() != best_checkpoint_path_) {
                    std::filesystem::remove(checkpoint_files[i]);
                    std::cout << "Removed old checkpoint: " << checkpoint_files[i].filename() << "\n";
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error during checkpoint cleanup: " << e.what() << "\n";
    }
}

void CheckpointCallback::LogCheckpoint(const std::string& path, int64_t step, float metric) const {
    std::cout << "Checkpoint saved: " << std::filesystem::path(path).filename().string()
              << " (step: " << step << ", " << config_.metric_name << ": " << metric << ")\n";
    
    if (path == best_checkpoint_path_) {
        std::cout << "  ^ New best checkpoint!\n";
    }
}

} // namespace training