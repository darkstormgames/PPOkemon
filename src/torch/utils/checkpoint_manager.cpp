#include "torch/utils/checkpoint_manager.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <sstream>

namespace utils {

CheckpointManager::CheckpointManager(const std::string& checkpoint_dir, 
                                   int max_to_keep,
                                   const std::string& metric_name)
    : checkpoint_dir_(checkpoint_dir), max_to_keep_(max_to_keep), 
      metric_name_(metric_name), save_only_best_(false),
      higher_is_better_(metric_name == "reward" || metric_name == "accuracy" || 
                       metric_name == "score") {
    CreateCheckpointDirectory();
    LoadCheckpointRegistry();
}

void CheckpointManager::CreateCheckpointDirectory() {
    std::filesystem::create_directories(checkpoint_dir_);
}

std::string CheckpointManager::GenerateCheckpointPath(int64_t step) const {
    std::ostringstream oss;
    oss << checkpoint_dir_ << "/checkpoint_" << std::setfill('0') << std::setw(8) << step;
    return oss.str();
}

std::string CheckpointManager::SaveCheckpoint(std::shared_ptr<torch::nn::Module> model, 
                                            torch::optim::Optimizer& optimizer,
                                            int64_t step,
                                            float metric_value,
                                            const std::unordered_map<std::string, float>& additional_metrics) {
    std::string checkpoint_path = GenerateCheckpointPath(step);
    std::string model_path = checkpoint_path + "_model.pt";
    std::string optimizer_path = checkpoint_path + "_optimizer.pt";
    std::string metadata_path = checkpoint_path + "_metadata.json";
    
    try {
        // Save model
        torch::save(model, model_path);
        
        // Save optimizer
        torch::save(optimizer, optimizer_path);
        
        // Save metadata
        SaveMetadata(metadata_path, step, metric_value, additional_metrics);
        
        // Add to checkpoint registry
        auto timestamp = std::chrono::system_clock::now();
        checkpoints_.emplace_back(model_path, optimizer_path, metadata_path, 
                                step, metric_value, timestamp);
        
        // Sort checkpoints by metric
        SortCheckpointsByMetric();
        
        // Clean up old checkpoints if necessary
        if (!save_only_best_) {
            CleanupOldCheckpoints();
        }
        
        // Save checkpoint registry
        SaveCheckpointRegistry();
        
        std::cout << "Checkpoint saved: " << checkpoint_path 
                  << " (step: " << step << ", " << metric_name_ << ": " << metric_value << ")" << std::endl;
        
        return checkpoint_path;
    } catch (const std::exception& e) {
        std::cerr << "Error saving checkpoint: " << e.what() << std::endl;
        return "";
    }
}

bool CheckpointManager::LoadCheckpoint(const std::string& checkpoint_path,
                                     std::shared_ptr<torch::nn::Module> model,
                                     torch::optim::Optimizer& optimizer,
                                     int64_t& step,
                                     float& metric_value) {
    std::string model_path = checkpoint_path + "_model.pt";
    std::string optimizer_path = checkpoint_path + "_optimizer.pt";
    std::string metadata_path = checkpoint_path + "_metadata.json";
    
    try {
        // Check if files exist
        if (!std::filesystem::exists(model_path) || 
            !std::filesystem::exists(optimizer_path) ||
            !std::filesystem::exists(metadata_path)) {
            std::cerr << "Checkpoint files not found: " << checkpoint_path << std::endl;
            return false;
        }
        
        // Load model
        torch::load(model, model_path);
        
        // Load optimizer
        torch::load(optimizer, optimizer_path);
        
        // Load metadata
        if (!LoadMetadata(metadata_path, step, metric_value)) {
            std::cerr << "Failed to load metadata: " << metadata_path << std::endl;
            return false;
        }
        
        std::cout << "Checkpoint loaded: " << checkpoint_path 
                  << " (step: " << step << ", " << metric_name_ << ": " << metric_value << ")" << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading checkpoint: " << e.what() << std::endl;
        return false;
    }
}

bool CheckpointManager::LoadLatestCheckpoint(std::shared_ptr<torch::nn::Module> model,
                                           torch::optim::Optimizer& optimizer,
                                           int64_t& step,
                                           float& metric_value) {
    if (checkpoints_.empty()) {
        std::cout << "No checkpoints found." << std::endl;
        return false;
    }
    
    // Find latest checkpoint by step
    auto latest_it = std::max_element(checkpoints_.begin(), checkpoints_.end(),
        [](const CheckpointData& a, const CheckpointData& b) {
            return a.step < b.step;
        });
    
    std::string checkpoint_path = latest_it->model_path;
    checkpoint_path = checkpoint_path.substr(0, checkpoint_path.find("_model.pt"));
    
    return LoadCheckpoint(checkpoint_path, model, optimizer, step, metric_value);
}

bool CheckpointManager::LoadBestCheckpoint(std::shared_ptr<torch::nn::Module> model,
                                         torch::optim::Optimizer& optimizer,
                                         int64_t& step,
                                         float& metric_value) {
    if (checkpoints_.empty()) {
        std::cout << "No checkpoints found." << std::endl;
        return false;
    }
    
    // Checkpoints are already sorted by metric
    const auto& best_checkpoint = checkpoints_[0];
    
    std::string checkpoint_path = best_checkpoint.model_path;
    checkpoint_path = checkpoint_path.substr(0, checkpoint_path.find("_model.pt"));
    
    return LoadCheckpoint(checkpoint_path, model, optimizer, step, metric_value);
}

std::vector<std::string> CheckpointManager::ListCheckpoints() const {
    std::vector<std::string> paths;
    for (const auto& checkpoint : checkpoints_) {
        std::string path = checkpoint.model_path;
        path = path.substr(0, path.find("_model.pt"));
        paths.push_back(path);
    }
    return paths;
}

void CheckpointManager::CleanupOldCheckpoints() {
    if (static_cast<int>(checkpoints_.size()) <= max_to_keep_) {
        return;
    }
    
    // Keep only the best max_to_keep_ checkpoints
    auto checkpoints_to_remove = checkpoints_.begin() + max_to_keep_;
    
    for (auto it = checkpoints_to_remove; it != checkpoints_.end(); ++it) {
        try {
            std::filesystem::remove(it->model_path);
            std::filesystem::remove(it->optimizer_path);
            std::filesystem::remove(it->metadata_path);
            std::cout << "Removed old checkpoint: " << it->model_path << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error removing checkpoint: " << e.what() << std::endl;
        }
    }
    
    checkpoints_.erase(checkpoints_to_remove, checkpoints_.end());
}

void CheckpointManager::DeleteCheckpoint(const std::string& checkpoint_path) {
    std::string model_path = checkpoint_path + "_model.pt";
    std::string optimizer_path = checkpoint_path + "_optimizer.pt";
    std::string metadata_path = checkpoint_path + "_metadata.json";
    
    try {
        std::filesystem::remove(model_path);
        std::filesystem::remove(optimizer_path);
        std::filesystem::remove(metadata_path);
        
        // Remove from registry
        checkpoints_.erase(std::remove_if(checkpoints_.begin(), checkpoints_.end(),
            [&model_path](const CheckpointData& checkpoint) {
                return checkpoint.model_path == model_path;
            }), checkpoints_.end());
        
        SaveCheckpointRegistry();
        std::cout << "Deleted checkpoint: " << checkpoint_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error deleting checkpoint: " << e.what() << std::endl;
    }
}

void CheckpointManager::DeleteAllCheckpoints() {
    for (const auto& checkpoint : checkpoints_) {
        try {
            std::filesystem::remove(checkpoint.model_path);
            std::filesystem::remove(checkpoint.optimizer_path);
            std::filesystem::remove(checkpoint.metadata_path);
        } catch (const std::exception& e) {
            std::cerr << "Error deleting checkpoint: " << e.what() << std::endl;
        }
    }
    
    checkpoints_.clear();
    SaveCheckpointRegistry();
    std::cout << "All checkpoints deleted." << std::endl;
}

bool CheckpointManager::HasCheckpoints() const {
    return !checkpoints_.empty();
}

std::string CheckpointManager::GetLatestCheckpointPath() const {
    if (checkpoints_.empty()) {
        return "";
    }
    
    auto latest_it = std::max_element(checkpoints_.begin(), checkpoints_.end(),
        [](const CheckpointData& a, const CheckpointData& b) {
            return a.step < b.step;
        });
    
    std::string path = latest_it->model_path;
    return path.substr(0, path.find("_model.pt"));
}

std::string CheckpointManager::GetBestCheckpointPath() const {
    if (checkpoints_.empty()) {
        return "";
    }
    
    // First checkpoint is the best (sorted)
    std::string path = checkpoints_[0].model_path;
    return path.substr(0, path.find("_model.pt"));
}

void CheckpointManager::PrintCheckpointStats() const {
    std::cout << "\n=== Checkpoint Statistics ===" << std::endl;
    std::cout << "Checkpoint directory: " << checkpoint_dir_ << std::endl;
    std::cout << "Number of checkpoints: " << checkpoints_.size() << std::endl;
    std::cout << "Max to keep: " << max_to_keep_ << std::endl;
    std::cout << "Metric name: " << metric_name_ << std::endl;
    std::cout << "Higher is better: " << (higher_is_better_ ? "Yes" : "No") << std::endl;
    
    if (!checkpoints_.empty()) {
        std::cout << "\nBest checkpoint:" << std::endl;
        const auto& best = checkpoints_[0];
        std::cout << "  Step: " << best.step << std::endl;
        std::cout << "  " << metric_name_ << ": " << best.metric_value << std::endl;
        
        auto latest_it = std::max_element(checkpoints_.begin(), checkpoints_.end(),
            [](const CheckpointData& a, const CheckpointData& b) {
                return a.step < b.step;
            });
        
        std::cout << "\nLatest checkpoint:" << std::endl;
        std::cout << "  Step: " << latest_it->step << std::endl;
        std::cout << "  " << metric_name_ << ": " << latest_it->metric_value << std::endl;
    }
    std::cout << "=============================" << std::endl;
}

void CheckpointManager::LoadCheckpointRegistry() {
    std::string registry_path = checkpoint_dir_ + "/checkpoint_registry.txt";
    
    if (!std::filesystem::exists(registry_path)) {
        return;
    }
    
    std::ifstream file(registry_path);
    std::string line;
    
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string model_path, optimizer_path, metadata_path;
        int64_t step;
        float metric_value;
        
        if (ss >> model_path >> optimizer_path >> metadata_path >> step >> metric_value) {
            // Check if files still exist
            if (std::filesystem::exists(model_path) && 
                std::filesystem::exists(optimizer_path) &&
                std::filesystem::exists(metadata_path)) {
                
                auto timestamp = std::chrono::system_clock::now(); // Approximate timestamp
                checkpoints_.emplace_back(model_path, optimizer_path, metadata_path,
                                        step, metric_value, timestamp);
            }
        }
    }
    
    SortCheckpointsByMetric();
}

void CheckpointManager::SaveCheckpointRegistry() {
    std::string registry_path = checkpoint_dir_ + "/checkpoint_registry.txt";
    std::ofstream file(registry_path);
    
    for (const auto& checkpoint : checkpoints_) {
        file << checkpoint.model_path << " " 
             << checkpoint.optimizer_path << " "
             << checkpoint.metadata_path << " "
             << checkpoint.step << " "
             << checkpoint.metric_value << std::endl;
    }
}

void CheckpointManager::SortCheckpointsByMetric() {
    std::sort(checkpoints_.begin(), checkpoints_.end(),
        [this](const CheckpointData& a, const CheckpointData& b) {
            if (higher_is_better_) {
                return a.metric_value > b.metric_value; // Best first
            } else {
                return a.metric_value < b.metric_value; // Best first (lowest for loss)
            }
        });
}

bool CheckpointManager::IsMetricBetter(float new_metric, float current_best) const {
    if (higher_is_better_) {
        return new_metric > current_best;
    } else {
        return new_metric < current_best;
    }
}

void CheckpointManager::SaveMetadata(const std::string& metadata_path, int64_t step, float metric_value,
                                   const std::unordered_map<std::string, float>& additional_metrics) {
    std::ofstream file(metadata_path);
    
    file << "{\n";
    file << "  \"step\": " << step << ",\n";
    file << "  \"" << metric_name_ << "\": " << metric_value;
    
    for (const auto& [key, value] : additional_metrics) {
        file << ",\n  \"" << key << "\": " << value;
    }
    
    file << "\n}\n";
}

bool CheckpointManager::LoadMetadata(const std::string& metadata_path, int64_t& step, float& metric_value) {
    std::ifstream file(metadata_path);
    if (!file.is_open()) {
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("\"step\"") != std::string::npos) {
            size_t colon_pos = line.find(':');
            if (colon_pos != std::string::npos) {
                std::string value_str = line.substr(colon_pos + 1);
                value_str.erase(0, value_str.find_first_not_of(" \t"));
                value_str.erase(value_str.find_last_not_of(" \t,") + 1);
                step = std::stoll(value_str);
            }
        } else if (line.find("\"" + metric_name_ + "\"") != std::string::npos) {
            size_t colon_pos = line.find(':');
            if (colon_pos != std::string::npos) {
                std::string value_str = line.substr(colon_pos + 1);
                value_str.erase(0, value_str.find_first_not_of(" \t"));
                value_str.erase(value_str.find_last_not_of(" \t,") + 1);
                metric_value = std::stof(value_str);
            }
        }
    }
    
    return true;
}

} // namespace utils
