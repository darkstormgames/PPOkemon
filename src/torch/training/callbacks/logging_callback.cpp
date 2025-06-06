// filepath: /home/timo/PPOkemon/src/torch/training/callbacks/logging_callback.cpp

#include "torch/training/callbacks/logging_callback.h"
#include "torch/utils/logger.h"
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace training {

LoggingCallback::LoggingCallback(const Config& config) 
    : config_(config), total_steps_(0) {
    
    // Create log directory
    std::filesystem::create_directories(config_.log_dir);
    
    // Initialize logger with simple constructor
    logger_ = std::make_unique<utils::TrainingLogger>(
        config_.log_dir, config_.experiment_name
    );
    
    // Reserve space for recent metrics
    recent_rewards_.reserve(1000);
    recent_losses_.reserve(1000);
}

// ============================================================================
// Callback Interface Implementation
// ============================================================================

void LoggingCallback::OnTrainBegin() {
    train_start_time_ = std::chrono::system_clock::now();
    last_log_time_ = train_start_time_;
    total_steps_ = 0;
    
    recent_rewards_.clear();
    recent_losses_.clear();
    
    if (config_.verbose) {
        std::cout << "\n=== Training Started ===\n";
        std::cout << "Experiment: " << config_.experiment_name << "\n";
        std::cout << "Log Directory: " << config_.log_dir << "\n";
        std::cout << "Log Interval: " << config_.log_interval << " steps\n";
        std::cout << "Progress Interval: " << config_.progress_interval << " steps\n";
        std::cout << "========================\n\n";
    }
    
    logger_->LogText("Training session started");
}

void LoggingCallback::OnStepEnd(int64_t step, float reward, float loss) {
    total_steps_ = step;
    
    // Log to our internal logger
    logger_->LogScalar("reward", reward, step);
    logger_->LogScalar("loss", loss, step);
    
    // Keep recent metrics for averaging
    recent_rewards_.push_back(reward);
    recent_losses_.push_back(loss);
    
    // Limit the size of recent metrics
    if (recent_rewards_.size() > 1000) {
        recent_rewards_.erase(recent_rewards_.begin());
    }
    if (recent_losses_.size() > 1000) {
        recent_losses_.erase(recent_losses_.begin());
    }
    
    // Log training statistics at intervals
    if (config_.verbose && step % config_.log_interval == 0) {
        LogTrainingStats(step, reward, loss);
    }
    
    // Log model statistics if enabled
    if ((config_.log_gradients || config_.log_weights) && step % config_.log_interval == 0) {
        LogModelStats(step);
    }
    
    // Show progress at intervals
    if (config_.verbose && step % config_.progress_interval == 0) {
        // We don't know total steps here, so just show current step
        std::cout << "Step " << step << " | Reward: " << std::fixed << std::setprecision(3) 
                  << reward << " | Loss: " << loss << "\n";
    }
}

void LoggingCallback::OnEvaluation(int64_t step, float eval_reward) {
    logger_->LogScalar("eval_reward", eval_reward, step);
    
    if (config_.verbose) {
        std::cout << "Evaluation at step " << step << " | Reward: " 
                  << std::fixed << std::setprecision(3) << eval_reward << "\n";
    }
}

void LoggingCallback::OnTrainEnd() {
    auto train_end_time = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(train_end_time - train_start_time_);
    
    if (config_.verbose) {
        std::cout << "\n=== Training Completed ===\n";
        std::cout << "Total Steps: " << total_steps_ << "\n";
        std::cout << "Training Time: " << FormatTime(duration) << "\n";
        std::cout << "Steps/Second: " << std::fixed << std::setprecision(2) 
                  << CalculateStepsPerSecond(total_steps_, duration) << "\n";
        
        if (!recent_rewards_.empty()) {
            float avg_reward = std::accumulate(recent_rewards_.end() - std::min(recent_rewards_.size(), 100UL), 
                                             recent_rewards_.end(), 0.0f) / 
                              std::min(recent_rewards_.size(), 100UL);
            std::cout << "Final Average Reward (last 100): " << std::fixed << std::setprecision(3) 
                      << avg_reward << "\n";
        }
        
        std::cout << "==========================\n\n";
    }
    
    // Save final logs
    SaveLogs();
}

// ============================================================================
// Logging Interface
// ============================================================================

void LoggingCallback::LogScalar(const std::string& name, float value, int64_t step) {
    logger_->LogScalar(name, value, step);
}

void LoggingCallback::LogScalars(const std::unordered_map<std::string, float>& scalars, int64_t step) {
    for (const auto& [name, value] : scalars) {
        logger_->LogScalar(name, value, step);
    }
}

void LoggingCallback::LogText(const std::string& message) {
    if (config_.verbose) {
        std::cout << "[LOG] " << message << "\n";
    }
    // Could also log to file if needed
}

void LoggingCallback::LogConfig(const std::unordered_map<std::string, std::string>& config) {
    logger_->LogConfig(config);
    
    if (config_.verbose) {
        std::cout << "\n=== Configuration ===\n";
        for (const auto& [key, value] : config) {
            std::cout << key << ": " << value << "\n";
        }
        std::cout << "=====================\n\n";
    }
}

// ============================================================================
// Statistics
// ============================================================================

float LoggingCallback::GetLatestMetric(const std::string& name) const {
    if (name == "reward" && !recent_rewards_.empty()) {
        return recent_rewards_.back();
    } else if (name == "loss" && !recent_losses_.empty()) {
        return recent_losses_.back();
    }
    return 0.0f;
}

float LoggingCallback::GetAverageMetric(const std::string& name, int last_n) const {
    if (name == "reward" && !recent_rewards_.empty()) {
        size_t n = std::min(recent_rewards_.size(), static_cast<size_t>(last_n));
        return std::accumulate(recent_rewards_.end() - n, recent_rewards_.end(), 0.0f) / n;
    } else if (name == "loss" && !recent_losses_.empty()) {
        size_t n = std::min(recent_losses_.size(), static_cast<size_t>(last_n));
        return std::accumulate(recent_losses_.end() - n, recent_losses_.end(), 0.0f) / n;
    }
    return 0.0f;
}

// ============================================================================
// Helper Methods
// ============================================================================

void LoggingCallback::LogProgress(int64_t current_step, int64_t total_steps) const {
    if (total_steps <= 0) return;
    
    float progress = static_cast<float>(current_step) / total_steps;
    int bar_length = 50;
    int filled_length = static_cast<int>(progress * bar_length);
    
    std::cout << "Progress: [";
    for (int i = 0; i < bar_length; ++i) {
        if (i < filled_length) {
            std::cout << "=";
        } else if (i == filled_length) {
            std::cout << ">";
        } else {
            std::cout << " ";
        }
    }
    std::cout << "] " << std::fixed << std::setprecision(1) << (progress * 100.0f) 
              << "% (" << current_step << "/" << total_steps << ")\n";
}

void LoggingCallback::LogTrainingStats(int64_t step, float reward, float loss) {
    auto current_time = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - last_log_time_);
    
    float avg_reward = GetAverageMetric("reward", 100);
    float avg_loss = GetAverageMetric("loss", 100);
    
    std::cout << "Step " << step 
              << " | Reward: " << std::fixed << std::setprecision(3) << reward 
              << " (avg: " << avg_reward << ")"
              << " | Loss: " << std::fixed << std::setprecision(3) << loss 
              << " (avg: " << avg_loss << ")";
    
    if (duration.count() > 0) {
        float steps_per_sec = static_cast<float>(config_.log_interval) / duration.count();
        std::cout << " | " << std::fixed << std::setprecision(1) << steps_per_sec << " steps/s";
    }
    
    std::cout << "\n";
    
    last_log_time_ = current_time;
}

void LoggingCallback::LogModelStats(int64_t step) {
    if (config_.log_gradients && get_gradients_fn_) {
        auto gradients = get_gradients_fn_();
        if (!gradients.empty()) {
            float grad_norm = 0.0f;
            for (float grad : gradients) {
                grad_norm += grad * grad;
            }
            grad_norm = std::sqrt(grad_norm);
            logger_->LogScalar("gradient_norm", grad_norm, step);
        }
    }
    
    if (config_.log_weights && get_weights_fn_) {
        auto weights = get_weights_fn_();
        if (!weights.empty()) {
            float weight_mean = std::accumulate(weights.begin(), weights.end(), 0.0f) / weights.size();
            logger_->LogScalar("weight_mean", weight_mean, step);
            
            float weight_std = 0.0f;
            for (float w : weights) {
                weight_std += (w - weight_mean) * (w - weight_mean);
            }
            weight_std = std::sqrt(weight_std / weights.size());
            logger_->LogScalar("weight_std", weight_std, step);
        }
    }
}

void LoggingCallback::SaveLogs() const {
    try {
        logger_->SaveToCSV("metrics.csv");
        logger_->SaveToJSON("metrics.json");
        
        if (config_.verbose) {
            std::cout << "Logs saved to: " << config_.log_dir << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error saving logs: " << e.what() << "\n";
    }
}

std::string LoggingCallback::FormatTime(std::chrono::seconds duration) const {
    int hours = duration.count() / 3600;
    int minutes = (duration.count() % 3600) / 60;
    int seconds = duration.count() % 60;
    
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(2) << hours << ":"
        << std::setfill('0') << std::setw(2) << minutes << ":"
        << std::setfill('0') << std::setw(2) << seconds;
    
    return oss.str();
}

float LoggingCallback::CalculateStepsPerSecond(int64_t steps, std::chrono::seconds duration) const {
    if (duration.count() == 0) return 0.0f;
    return static_cast<float>(steps) / duration.count();
}

} // namespace training