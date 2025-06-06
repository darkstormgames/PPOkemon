#include "torch/training/trainer.h"
#include "torch/utils/logger.h"
#include "torch/utils/checkpoint_manager.h"
#include "torch/utils/lr_scheduler.h"
#include "torch/utils/profiler.h"
#include "torch/training/evaluation/evaluator.h"
#include "torch/training/evaluation/metrics.h"
#include "torch/training/callbacks/checkpoint_callback.h"
#include "torch/training/callbacks/logging_callback.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <filesystem>

namespace training {

// ============================================================================
// Constructor & Destructor
// ============================================================================

Trainer::Trainer(const Config& config)
    : config_(config)
    , current_step_(0)
    , current_reward_(0.0f)
    , best_reward_(-std::numeric_limits<float>::infinity())
    , should_stop_(false)
    , steps_without_improvement_(0)
    , early_stopping_best_metric_(-std::numeric_limits<float>::infinity())
{
    InitializeTraining();
}

Trainer::~Trainer() {
    if (logger_) {
        logger_->LogText("Training session ended");
        logger_->Flush();
    }
}

// ============================================================================
// Training Control
// ============================================================================

float Trainer::GetLearningRate() const {
    if (lr_scheduler_) {
        return lr_scheduler_->GetLR();
    }
    return config_.learning_rate;
}

// ============================================================================
// Training Infrastructure
// ============================================================================

void Trainer::InitializeTraining() {
    try {
        // Create log directory
        std::filesystem::create_directories(config_.log_dir);
        std::filesystem::create_directories(config_.checkpoint_dir);
        
        // Initialize logger
        logger_ = std::make_unique<utils::TrainingLogger>(
            config_.log_dir, config_.experiment_name
        );
        logger_->SetVerbose(config_.verbose);
        logger_->SetLogInterval(config_.log_interval);
        
        // Initialize checkpoint manager
        checkpoint_manager_ = std::make_unique<utils::CheckpointManager>(
            config_.checkpoint_dir, 5, config_.early_stopping_metric
        );
        
        // Initialize learning rate scheduler (placeholder - will be set by subclasses)
        // lr_scheduler_ will be initialized by concrete implementations when optimizer is available
        
        // Log configuration
        std::unordered_map<std::string, std::string> config_map = {
            {"total_steps", std::to_string(config_.total_steps)},
            {"batch_size", std::to_string(config_.batch_size)},
            {"learning_rate", std::to_string(config_.learning_rate)},
            {"num_envs", std::to_string(config_.num_envs)},
            {"max_episode_steps", std::to_string(config_.max_episode_steps)},
            {"eval_interval", std::to_string(config_.eval_interval)},
            {"save_interval", std::to_string(config_.save_interval)},
            {"log_interval", std::to_string(config_.log_interval)},
            {"gradient_clip_norm", std::to_string(config_.gradient_clip_norm)},
            {"device", config_.device.str()},
            {"experiment_name", config_.experiment_name},
            {"early_stopping", config_.use_early_stopping ? "true" : "false"},
            {"early_stopping_patience", std::to_string(config_.early_stopping_patience)},
            {"early_stopping_metric", config_.early_stopping_metric}
        };
        
        logger_->LogConfig(config_map);
        logger_->LogText("Training initialized successfully");
        
        if (config_.verbose) {
            std::cout << "=== Training Configuration ===" << std::endl;
            std::cout << "Experiment: " << config_.experiment_name << std::endl;
            std::cout << "Device: " << config_.device.str() << std::endl;
            std::cout << "Total steps: " << config_.total_steps << std::endl;
            std::cout << "Batch size: " << config_.batch_size << std::endl;
            std::cout << "Learning rate: " << config_.learning_rate << std::endl;
            std::cout << "Environments: " << config_.num_envs << std::endl;
            std::cout << "Log directory: " << config_.log_dir << std::endl;
            std::cout << "Checkpoint directory: " << config_.checkpoint_dir << std::endl;
            std::cout << "===============================" << std::endl;
        }
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize training: " + std::string(e.what()));
    }
}

void Trainer::LogMetrics(const std::unordered_map<std::string, float>& metrics) {
    if (!logger_) return;
    
    // Log all metrics
    logger_->LogScalars(metrics, current_step_);
    
    // Update current reward if present
    auto reward_it = metrics.find("reward");
    if (reward_it != metrics.end()) {
        current_reward_ = reward_it->second;
        
        // Update best reward
        if (current_reward_ > best_reward_) {
            best_reward_ = current_reward_;
            if (config_.verbose && current_step_ % config_.log_interval == 0) {
                std::cout << "New best reward: " << best_reward_ << " at step " << current_step_ << std::endl;
            }
        }
    }
    
    // Call step callbacks
    float loss = 0.0f;
    auto loss_it = metrics.find("loss");
    if (loss_it != metrics.end()) {
        loss = loss_it->second;
    }
    
    for (const auto& callback : step_callbacks_) {
        callback(current_step_, current_reward_, loss);
    }
    
    // Log progress periodically
    if (config_.verbose && current_step_ % config_.log_interval == 0) {
        logger_->PrintProgress(current_step_, config_.total_steps);
    }
}

void Trainer::CheckpointModel(float metric_value) {
    if (!checkpoint_manager_) return;
    
    try {
        // This is a placeholder - concrete implementations should override this
        // and save their specific model and optimizer states
        
        if (config_.verbose) {
            std::cout << "Checkpoint requested with metric value: " << metric_value 
                      << " at step " << current_step_ << std::endl;
        }
        
        logger_->LogText("Checkpoint saved with " + config_.early_stopping_metric + 
                        ": " + std::to_string(metric_value));
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to save checkpoint: " << e.what() << std::endl;
        if (logger_) {
            logger_->LogText("Checkpoint save failed: " + std::string(e.what()));
        }
    }
}

void Trainer::UpdateLearningRate() {
    if (lr_scheduler_) {
        lr_scheduler_->Step(current_step_);
        
        // Log learning rate changes
        float current_lr = lr_scheduler_->GetLR();
        logger_->LogScalar("learning_rate", current_lr, current_step_);
    }
}

bool Trainer::CheckEarlyStopping() {
    if (!config_.use_early_stopping) {
        return false;
    }
    
    // Check if we have improvement in the target metric
    bool has_improvement = false;
    
    if (config_.early_stopping_metric == "reward") {
        if (current_reward_ > early_stopping_best_metric_) {
            early_stopping_best_metric_ = current_reward_;
            steps_without_improvement_ = 0;
            has_improvement = true;
        }
    } else {
        // For other metrics, assume lower is better (like loss)
        // Concrete implementations should override this logic if needed
        if (logger_) {
            const auto& metric_data = logger_->GetMetric(config_.early_stopping_metric);
            if (!metric_data.values.empty()) {
                float current_metric = metric_data.GetLatest();
                if (current_metric < early_stopping_best_metric_ || 
                    early_stopping_best_metric_ == -std::numeric_limits<float>::infinity()) {
                    early_stopping_best_metric_ = current_metric;
                    steps_without_improvement_ = 0;
                    has_improvement = true;
                }
            }
        }
    }
    
    if (!has_improvement) {
        steps_without_improvement_++;
    }
    
    bool should_stop = steps_without_improvement_ >= config_.early_stopping_patience;
    
    if (should_stop && config_.verbose) {
        std::cout << "Early stopping triggered after " << steps_without_improvement_ 
                  << " steps without improvement in " << config_.early_stopping_metric << std::endl;
        
        if (logger_) {
            logger_->LogText("Early stopping triggered - no improvement in " + 
                           config_.early_stopping_metric + " for " + 
                           std::to_string(steps_without_improvement_) + " steps");
        }
    }
    
    return should_stop;
}

// ============================================================================
// Training Loop Template Methods (to be implemented by subclasses)
// ============================================================================

// These methods are pure virtual and must be implemented by concrete algorithm classes:
// - Train()
// - Evaluate() 
// - SaveModel()
// - LoadModel()
// - CollectExperience()
// - UpdatePolicy()
// - ResetEnvironments()

} // namespace training