#pragma once

#include "checkpoint_callback.h"
#include <chrono>
#include <vector>

namespace utils {
    class TrainingLogger;
}

namespace training {

/**
 * Logging callback for training metrics and progress
 */
class LoggingCallback : public Callback {
public:
    struct Config {
        std::string log_dir = "./logs";
        std::string experiment_name = "training";
        int log_interval = 1000;        // Steps between console logs
        int progress_interval = 10000;  // Steps between progress updates
        bool verbose = true;
        bool log_gradients = false;
        bool log_weights = false;
        bool save_csv = true;
        bool save_json = true;
    };

    LoggingCallback(const Config& config);
    ~LoggingCallback() = default;

    // Callback interface
    void OnTrainBegin() override;
    void OnStepEnd(int64_t step, float reward, float loss) override;
    void OnEvaluation(int64_t step, float eval_reward) override;
    void OnTrainEnd() override;

    // Logging interface
    void LogScalar(const std::string& name, float value, int64_t step);
    void LogScalars(const std::unordered_map<std::string, float>& scalars, int64_t step);
    void LogText(const std::string& message);
    void LogConfig(const std::unordered_map<std::string, std::string>& config);

    // Model logging
    using GetGradientsFunction = std::function<std::vector<float>()>;
    using GetWeightsFunction = std::function<std::vector<float>()>;
    
    void SetGradientsFunction(const GetGradientsFunction& grad_fn) { get_gradients_fn_ = grad_fn; }
    void SetWeightsFunction(const GetWeightsFunction& weights_fn) { get_weights_fn_ = weights_fn; }

    // Configuration
    void SetLogInterval(int interval) { config_.log_interval = interval; }
    void SetVerbose(bool verbose) { config_.verbose = verbose; }
    void SetLogGradients(bool log_gradients) { config_.log_gradients = log_gradients; }

    // Statistics
    float GetLatestMetric(const std::string& name) const;
    float GetAverageMetric(const std::string& name, int last_n = 100) const;

private:
    Config config_;
    std::unique_ptr<utils::TrainingLogger> logger_;
    
    GetGradientsFunction get_gradients_fn_;
    GetWeightsFunction get_weights_fn_;
    
    // Timing
    std::chrono::system_clock::time_point train_start_time_;
    std::chrono::system_clock::time_point last_log_time_;
    
    // Statistics
    std::vector<float> recent_rewards_;
    std::vector<float> recent_losses_;
    int64_t total_steps_;
    
    // Helper methods
    void LogProgress(int64_t current_step, int64_t total_steps) const;
    void LogTrainingStats(int64_t step, float reward, float loss);
    void LogModelStats(int64_t step);
    void SaveLogs() const;
    std::string FormatTime(std::chrono::seconds duration) const;
    float CalculateStepsPerSecond(int64_t steps, std::chrono::seconds duration) const;
};

} // namespace training
