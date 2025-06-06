#pragma once

#include <string>
#include <chrono>
#include <unordered_map>
#include <vector>
#include <memory>

namespace utils {

class Timer {
public:
    Timer();
    ~Timer() = default;
    
    // Basic timing
    void Start();
    void Stop();
    void Reset();
    
    // Get elapsed time
    double GetElapsedSeconds() const;
    double GetElapsedMilliseconds() const;
    double GetElapsedMicroseconds() const;
    
    // Check if timer is running
    bool IsRunning() const { return is_running_; }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool is_running_;
};

class ScopedTimer {
public:
    ScopedTimer(const std::string& name);
    ~ScopedTimer();
    
private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

class Profiler {
public:
    static Profiler& Instance();
    
    // Manual timing
    void StartTimer(const std::string& name);
    void StopTimer(const std::string& name);
    
    // Scoped timing
    std::unique_ptr<ScopedTimer> TimeScope(const std::string& name);
    
    // Record single measurement
    void RecordTime(const std::string& name, double time_seconds);
    
    // Get statistics
    double GetAverageTime(const std::string& name) const;
    double GetTotalTime(const std::string& name) const;
    int64_t GetCallCount(const std::string& name) const;
    double GetMinTime(const std::string& name) const;
    double GetMaxTime(const std::string& name) const;
    
    // Reset and clear
    void Reset();
    void Clear();
    
    // Reporting
    void PrintReport(bool sort_by_total_time = true) const;
    void SaveReport(const std::string& filename) const;
    
    // Enable/disable profiling
    void SetEnabled(bool enabled) { enabled_ = enabled; }
    bool IsEnabled() const { return enabled_; }

private:
    struct TimingData {
        std::vector<double> times;
        double total_time = 0.0;
        double min_time = std::numeric_limits<double>::max();
        double max_time = 0.0;
        int64_t call_count = 0;
        std::chrono::high_resolution_clock::time_point start_time;
        bool is_running = false;
        
        void AddTime(double time) {
            times.push_back(time);
            total_time += time;
            min_time = std::min(min_time, time);
            max_time = std::max(max_time, time);
            call_count++;
        }
        
        double GetAverage() const {
            return call_count > 0 ? total_time / call_count : 0.0;
        }
    };
    
    Profiler() = default;
    
    std::unordered_map<std::string, TimingData> timing_data_;
    bool enabled_ = true;
};

// Convenience macros
#define PROFILE_SCOPE(name) \
    auto timer = utils::Profiler::Instance().TimeScope(name)

#define PROFILE_FUNCTION() \
    PROFILE_SCOPE(__FUNCTION__)

#define START_TIMER(name) \
    utils::Profiler::Instance().StartTimer(name)

#define STOP_TIMER(name) \
    utils::Profiler::Instance().StopTimer(name)

} // namespace utils
