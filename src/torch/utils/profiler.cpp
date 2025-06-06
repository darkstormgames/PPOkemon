#include "torch/utils/profiler.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <limits>

namespace utils {

// Timer implementation
Timer::Timer() : is_running_(false) {
    Reset();
}

void Timer::Start() {
    start_time_ = std::chrono::high_resolution_clock::now();
    is_running_ = true;
}

void Timer::Stop() {
    if (is_running_) {
        end_time_ = std::chrono::high_resolution_clock::now();
        is_running_ = false;
    }
}

void Timer::Reset() {
    start_time_ = std::chrono::high_resolution_clock::now();
    end_time_ = start_time_;
    is_running_ = false;
}

double Timer::GetElapsedSeconds() const {
    auto end_point = is_running_ ? std::chrono::high_resolution_clock::now() : end_time_;
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_point - start_time_);
    return duration.count() / 1e9;
}

double Timer::GetElapsedMilliseconds() const {
    return GetElapsedSeconds() * 1000.0;
}

double Timer::GetElapsedMicroseconds() const {
    return GetElapsedSeconds() * 1e6;
}

// ScopedTimer implementation
ScopedTimer::ScopedTimer(const std::string& name) 
    : name_(name), start_time_(std::chrono::high_resolution_clock::now()) {
}

ScopedTimer::~ScopedTimer() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time_);
    double time_seconds = duration.count() / 1e9;
    
    Profiler::Instance().RecordTime(name_, time_seconds);
}

// Profiler implementation
Profiler& Profiler::Instance() {
    static Profiler instance;
    return instance;
}

void Profiler::StartTimer(const std::string& name) {
    if (!enabled_) return;
    
    auto& data = timing_data_[name];
    if (data.is_running) {
        std::cerr << "Warning: Timer '" << name << "' is already running" << std::endl;
        return;
    }
    
    data.start_time = std::chrono::high_resolution_clock::now();
    data.is_running = true;
}

void Profiler::StopTimer(const std::string& name) {
    if (!enabled_) return;
    
    auto it = timing_data_.find(name);
    if (it == timing_data_.end()) {
        std::cerr << "Warning: Timer '" << name << "' was never started" << std::endl;
        return;
    }
    
    auto& data = it->second;
    if (!data.is_running) {
        std::cerr << "Warning: Timer '" << name << "' is not running" << std::endl;
        return;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - data.start_time);
    double time_seconds = duration.count() / 1e9;
    
    data.AddTime(time_seconds);
    data.is_running = false;
}

std::unique_ptr<ScopedTimer> Profiler::TimeScope(const std::string& name) {
    if (!enabled_) {
        return nullptr;
    }
    return std::make_unique<ScopedTimer>(name);
}

void Profiler::RecordTime(const std::string& name, double time_seconds) {
    if (!enabled_) return;
    
    auto& data = timing_data_[name];
    data.AddTime(time_seconds);
}

double Profiler::GetAverageTime(const std::string& name) const {
    auto it = timing_data_.find(name);
    if (it == timing_data_.end()) {
        return 0.0;
    }
    return it->second.GetAverage();
}

double Profiler::GetTotalTime(const std::string& name) const {
    auto it = timing_data_.find(name);
    if (it == timing_data_.end()) {
        return 0.0;
    }
    return it->second.total_time;
}

int64_t Profiler::GetCallCount(const std::string& name) const {
    auto it = timing_data_.find(name);
    if (it == timing_data_.end()) {
        return 0;
    }
    return it->second.call_count;
}

double Profiler::GetMinTime(const std::string& name) const {
    auto it = timing_data_.find(name);
    if (it == timing_data_.end()) {
        return 0.0;
    }
    return it->second.min_time == std::numeric_limits<double>::max() ? 0.0 : it->second.min_time;
}

double Profiler::GetMaxTime(const std::string& name) const {
    auto it = timing_data_.find(name);
    if (it == timing_data_.end()) {
        return 0.0;
    }
    return it->second.max_time;
}

void Profiler::Reset() {
    for (auto& pair : timing_data_) {
        auto& data = pair.second;
        data.times.clear();
        data.total_time = 0.0;
        data.min_time = std::numeric_limits<double>::max();
        data.max_time = 0.0;
        data.call_count = 0;
        data.is_running = false;
    }
}

void Profiler::Clear() {
    timing_data_.clear();
}

void Profiler::PrintReport(bool sort_by_total_time) const {
    if (timing_data_.empty()) {
        std::cout << "No profiling data available." << std::endl;
        return;
    }
    
    // Create sorted list of entries
    std::vector<std::pair<std::string, const TimingData*>> entries;
    for (const auto& pair : timing_data_) {
        entries.emplace_back(pair.first, &pair.second);
    }
    
    if (sort_by_total_time) {
        std::sort(entries.begin(), entries.end(),
                 [](const auto& a, const auto& b) {
                     return a.second->total_time > b.second->total_time;
                 });
    } else {
        std::sort(entries.begin(), entries.end(),
                 [](const auto& a, const auto& b) {
                     return a.second->GetAverage() > b.second->GetAverage();
                 });
    }
    
    // Print header
    std::cout << "\n" << std::string(100, '=') << std::endl;
    std::cout << "PROFILING REPORT" << std::endl;
    std::cout << std::string(100, '=') << std::endl;
    std::cout << std::left << std::setw(30) << "Name"
              << std::right << std::setw(12) << "Calls"
              << std::setw(15) << "Total (ms)"
              << std::setw(15) << "Avg (ms)"
              << std::setw(15) << "Min (ms)"
              << std::setw(15) << "Max (ms)" << std::endl;
    std::cout << std::string(100, '-') << std::endl;
    
    // Print entries
    for (const auto& entry : entries) {
        const std::string& name = entry.first;
        const TimingData& data = *entry.second;
        
        std::cout << std::left << std::setw(30) << name
                  << std::right << std::setw(12) << data.call_count
                  << std::setw(15) << std::fixed << std::setprecision(3) << (data.total_time * 1000.0)
                  << std::setw(15) << std::fixed << std::setprecision(3) << (data.GetAverage() * 1000.0)
                  << std::setw(15) << std::fixed << std::setprecision(3) << (data.min_time * 1000.0)
                  << std::setw(15) << std::fixed << std::setprecision(3) << (data.max_time * 1000.0)
                  << std::endl;
    }
    
    std::cout << std::string(100, '=') << std::endl;
    
    // Calculate total time across all timers
    double total_time_all = 0.0;
    for (const auto& pair : timing_data_) {
        total_time_all += pair.second.total_time;
    }
    std::cout << "Total profiled time: " << std::fixed << std::setprecision(3) 
              << (total_time_all * 1000.0) << " ms" << std::endl;
    std::cout << std::string(100, '=') << std::endl << std::endl;
}

void Profiler::SaveReport(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Could not open file '" << filename << "' for writing" << std::endl;
        return;
    }
    
    // Redirect cout to file temporarily
    std::streambuf* orig_cout = std::cout.rdbuf();
    std::cout.rdbuf(file.rdbuf());
    
    PrintReport(true);
    
    // Restore cout
    std::cout.rdbuf(orig_cout);
    
    file.close();
    std::cout << "Profiling report saved to: " << filename << std::endl;
}

} // namespace utils
