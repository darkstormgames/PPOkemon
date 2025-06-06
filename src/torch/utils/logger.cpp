#include "torch/utils/logger.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace utils {

// Global logger instance
std::unique_ptr<TrainingLogger> g_logger = nullptr;

TrainingLogger::TrainingLogger(const std::string& log_dir, const std::string& experiment_name)
    : log_dir_(log_dir), experiment_name_(experiment_name), verbose_(true), 
      log_interval_(100), last_log_step_(0), start_time_(std::chrono::system_clock::now()) {
    
    CreateLogDirectory();
    
    // Create log file with timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << log_dir_ << "/" << experiment_name_ << "_" 
       << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".log";
    log_file_path_ = ss.str();
    
    log_file_.open(log_file_path_, std::ios::app);
    if (!log_file_.is_open()) {
        throw std::runtime_error("Failed to open log file: " + log_file_path_);
    }
    
    WriteToFile("=== Training Log Started ===");
    WriteToFile("Experiment: " + experiment_name_);
    WriteToFile("Start Time: " + GetTimestamp());
}

TrainingLogger::~TrainingLogger() {
    if (log_file_.is_open()) {
        WriteToFile("=== Training Log Ended ===");
        WriteToFile("End Time: " + GetTimestamp());
        log_file_.close();
    }
}

void TrainingLogger::LogScalar(const std::string& name, float value, int64_t step) {
    metrics_[name].Add(value, step);
    
    if (verbose_ && (step - last_log_step_) >= log_interval_) {
        std::cout << "Step " << step << " | " << name << ": " << value << std::endl;
        last_log_step_ = step;
    }
    
    // Write to log file
    std::stringstream ss;
    ss << "[" << GetTimestamp() << "] Step " << step << " | " << name << ": " << value;
    WriteToFile(ss.str());
}

void TrainingLogger::LogScalars(const std::unordered_map<std::string, float>& scalars, int64_t step) {
    for (const auto& pair : scalars) {
        metrics_[pair.first].Add(pair.second, step);
    }
    
    if (verbose_ && (step - last_log_step_) >= log_interval_) {
        std::cout << "Step " << step << " | ";
        for (const auto& pair : scalars) {
            std::cout << pair.first << ": " << pair.second << " ";
        }
        std::cout << std::endl;
        last_log_step_ = step;
    }
    
    // Write to log file
    std::stringstream ss;
    ss << "[" << GetTimestamp() << "] Step " << step << " | ";
    for (const auto& pair : scalars) {
        ss << pair.first << ": " << pair.second << " ";
    }
    WriteToFile(ss.str());
}

void TrainingLogger::LogHistogram(const std::string& name, const std::vector<float>& values, int64_t step) {
    if (values.empty()) return;
    
    // Calculate basic statistics
    float min_val = *std::min_element(values.begin(), values.end());
    float max_val = *std::max_element(values.begin(), values.end());
    float sum = std::accumulate(values.begin(), values.end(), 0.0f);
    float mean = sum / values.size();
    
    // Calculate standard deviation
    float variance = 0.0f;
    for (float val : values) {
        variance += (val - mean) * (val - mean);
    }
    variance /= values.size();
    float std_dev = std::sqrt(variance);
    
    std::stringstream ss;
    ss << "[" << GetTimestamp() << "] Step " << step << " | " << name << " histogram: "
       << "min=" << min_val << ", max=" << max_val << ", mean=" << mean 
       << ", std=" << std_dev << ", count=" << values.size();
    WriteToFile(ss.str());
    
    if (verbose_) {
        std::cout << "Step " << step << " | " << name << " histogram: "
                  << "min=" << min_val << ", max=" << max_val << ", mean=" << mean 
                  << ", std=" << std_dev << std::endl;
    }
}

void TrainingLogger::LogText(const std::string& message) {
    std::string log_msg = "[" + GetTimestamp() + "] " + message;
    WriteToFile(log_msg);
    
    if (verbose_) {
        std::cout << message << std::endl;
    }
}

void TrainingLogger::LogText(const std::string& tag, const std::string& message, int64_t step) {
    std::stringstream ss;
    ss << "[" << GetTimestamp() << "] Step " << step << " | " << tag << ": " << message;
    WriteToFile(ss.str());
    
    if (verbose_) {
        std::cout << "Step " << step << " | " << tag << ": " << message << std::endl;
    }
}

void TrainingLogger::LogModel(const std::string& model_path, int64_t step) {
    std::stringstream ss;
    ss << "[" << GetTimestamp() << "] Step " << step << " | Model saved: " << model_path;
    WriteToFile(ss.str());
    
    if (verbose_) {
        std::cout << "Step " << step << " | Model saved: " << model_path << std::endl;
    }
}

void TrainingLogger::LogConfig(const std::unordered_map<std::string, std::string>& config) {
    WriteToFile("=== Configuration ===");
    for (const auto& pair : config) {
        WriteToFile(pair.first + ": " + pair.second);
    }
    WriteToFile("====================");
    
    if (verbose_) {
        std::cout << "=== Configuration ===" << std::endl;
        for (const auto& pair : config) {
            std::cout << pair.first << ": " << pair.second << std::endl;
        }
        std::cout << "====================" << std::endl;
    }
}

const MetricData& TrainingLogger::GetMetric(const std::string& name) const {
    auto it = metrics_.find(name);
    if (it == metrics_.end()) {
        static MetricData empty_metric;
        return empty_metric;
    }
    return it->second;
}

std::vector<std::string> TrainingLogger::GetMetricNames() const {
    std::vector<std::string> names;
    names.reserve(metrics_.size());
    for (const auto& pair : metrics_) {
        names.push_back(pair.first);
    }
    return names;
}

void TrainingLogger::PrintSummary(int64_t last_n_steps) const {
    std::cout << "\n=== Training Summary ===" << std::endl;
    
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);
    std::cout << "Training Duration: " << FormatDuration(duration) << std::endl;
    
    for (const auto& pair : metrics_) {
        const auto& metric = pair.second;
        if (!metric.values.empty()) {
            size_t count = std::min(static_cast<size_t>(last_n_steps), metric.values.size());
            float avg = metric.GetAverage(count);
            float latest = metric.GetLatest();
            std::cout << pair.first << ": latest=" << latest << ", avg=" << avg 
                      << " (last " << count << " steps)" << std::endl;
        }
    }
    std::cout << "========================\n" << std::endl;
}

void TrainingLogger::PrintProgress(int64_t current_step, int64_t total_steps) const {
    float progress = static_cast<float>(current_step) / total_steps;
    int bar_width = 50;
    int filled = static_cast<int>(progress * bar_width);
    
    std::cout << "\rProgress: [";
    for (int i = 0; i < bar_width; ++i) {
        if (i < filled) std::cout << "=";
        else if (i == filled) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setprecision(1) 
              << (progress * 100.0f) << "% (" << current_step << "/" << total_steps << ")";
    std::cout.flush();
}

void TrainingLogger::SaveToCSV(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + filename);
    }
    
    // Write header
    file << "step";
    for (const auto& pair : metrics_) {
        file << "," << pair.first;
    }
    file << "\n";
    
    // Find maximum step count
    int64_t max_steps = 0;
    for (const auto& pair : metrics_) {
        if (!pair.second.steps.empty()) {
            max_steps = std::max(max_steps, pair.second.steps.back());
        }
    }
    
    // Write data row by row
    for (int64_t step = 0; step <= max_steps; ++step) {
        file << step;
        for (const auto& pair : metrics_) {
            const auto& metric = pair.second;
            float value = 0.0f;
            
            // Find value for this step
            auto it = std::find(metric.steps.begin(), metric.steps.end(), step);
            if (it != metric.steps.end()) {
                size_t index = std::distance(metric.steps.begin(), it);
                value = metric.values[index];
            }
            
            file << "," << value;
        }
        file << "\n";
    }
}

void TrainingLogger::SaveToJSON(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open JSON file: " + filename);
    }
    
    file << "{\n";
    file << "  \"experiment\": \"" << experiment_name_ << "\",\n";
    file << "  \"start_time\": \"" << GetTimestamp() << "\",\n";
    file << "  \"metrics\": {\n";
    
    bool first_metric = true;
    for (const auto& pair : metrics_) {
        if (!first_metric) file << ",\n";
        first_metric = false;
        
        file << "    \"" << pair.first << "\": {\n";
        file << "      \"steps\": [";
        for (size_t i = 0; i < pair.second.steps.size(); ++i) {
            if (i > 0) file << ", ";
            file << pair.second.steps[i];
        }
        file << "],\n";
        
        file << "      \"values\": [";
        for (size_t i = 0; i < pair.second.values.size(); ++i) {
            if (i > 0) file << ", ";
            file << pair.second.values[i];
        }
        file << "]\n";
        file << "    }";
    }
    
    file << "\n  }\n";
    file << "}\n";
}

void TrainingLogger::Flush() {
    if (log_file_.is_open()) {
        log_file_.flush();
    }
}

void TrainingLogger::CreateLogDirectory() {
    try {
        std::filesystem::create_directories(log_dir_);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to create log directory: " + log_dir_ + " - " + e.what());
    }
}

std::string TrainingLogger::GetTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << "." << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

void TrainingLogger::WriteToFile(const std::string& message) {
    if (log_file_.is_open()) {
        log_file_ << message << std::endl;
        log_file_.flush();
    }
}

std::string TrainingLogger::FormatDuration(std::chrono::seconds duration) const {
    auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration % std::chrono::hours(1));
    auto seconds = duration % std::chrono::minutes(1);
    
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(2) << hours.count() << ":"
       << std::setfill('0') << std::setw(2) << minutes.count() << ":"
       << std::setfill('0') << std::setw(2) << seconds.count();
    return ss.str();
}

} // namespace utils
