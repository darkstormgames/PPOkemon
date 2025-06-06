#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <memory>
#include <chrono>

namespace utils {

struct MetricData {
    std::vector<float> values;
    std::vector<int64_t> steps;
    std::vector<std::chrono::system_clock::time_point> timestamps;
    
    void Add(float value, int64_t step) {
        values.push_back(value);
        steps.push_back(step);
        timestamps.push_back(std::chrono::system_clock::now());
    }
    
    void Clear() {
        values.clear();
        steps.clear();
        timestamps.clear();
    }
    
    float GetLatest() const {
        return values.empty() ? 0.0f : values.back();
    }
    
    float GetAverage(size_t last_n = 0) const {
        if (values.empty()) return 0.0f;
        
        size_t start_idx = (last_n > 0 && last_n < values.size()) ? 
                          values.size() - last_n : 0;
        
        float sum = 0.0f;
        for (size_t i = start_idx; i < values.size(); ++i) {
            sum += values[i];
        }
        return sum / (values.size() - start_idx);
    }
};

class TrainingLogger {
public:
    TrainingLogger(const std::string& log_dir = "./logs", 
                   const std::string& experiment_name = "experiment");
    ~TrainingLogger();
    
    // Logging methods
    void LogScalar(const std::string& name, float value, int64_t step);
    void LogScalars(const std::unordered_map<std::string, float>& scalars, int64_t step);
    
    // Histogram logging (for weight distributions, etc.)
    void LogHistogram(const std::string& name, const std::vector<float>& values, int64_t step);
    
    // Text logging
    void LogText(const std::string& message);
    void LogText(const std::string& tag, const std::string& message, int64_t step);
    
    // Model checkpoint logging
    void LogModel(const std::string& model_path, int64_t step);
    
    // Configuration logging
    void LogConfig(const std::unordered_map<std::string, std::string>& config);
    
    // Metric access
    const MetricData& GetMetric(const std::string& name) const;
    std::vector<std::string> GetMetricNames() const;
    
    // Statistics
    void PrintSummary(int64_t last_n_steps = 100) const;
    void PrintProgress(int64_t current_step, int64_t total_steps) const;
    
    // File operations
    void SaveToCSV(const std::string& filename) const;
    void SaveToJSON(const std::string& filename) const;
    void Flush(); // Force write all buffered data
    
    // Console output control
    void SetVerbose(bool verbose) { verbose_ = verbose; }
    void SetLogInterval(int64_t interval) { log_interval_ = interval; }

private:
    std::string log_dir_;
    std::string experiment_name_;
    std::string log_file_path_;
    
    std::unordered_map<std::string, MetricData> metrics_;
    std::ofstream log_file_;
    
    bool verbose_;
    int64_t log_interval_;
    int64_t last_log_step_;
    
    std::chrono::system_clock::time_point start_time_;
    
    // Helper methods
    void CreateLogDirectory();
    std::string GetTimestamp() const;
    void WriteToFile(const std::string& message);
    std::string FormatDuration(std::chrono::seconds duration) const;
};

// Global logger instance for convenience
extern std::unique_ptr<TrainingLogger> g_logger;

// Convenience macros
#define LOG_SCALAR(name, value, step) \
    if (utils::g_logger) utils::g_logger->LogScalar(name, value, step)

#define LOG_TEXT(message) \
    if (utils::g_logger) utils::g_logger->LogText(message)

#define LOG_CONFIG(config) \
    if (utils::g_logger) utils::g_logger->LogConfig(config)

} // namespace utils
