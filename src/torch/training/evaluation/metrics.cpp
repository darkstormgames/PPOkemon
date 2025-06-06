// filepath: /home/timo/PPOkemon/src/torch/training/evaluation/metrics.cpp

#include "torch/training/evaluation/metrics.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace training {

// Static member definition
std::unordered_map<std::string, Metrics::CustomMetricFunction> Metrics::custom_metrics_;

// ============================================================================
// Basic Episode Statistics
// ============================================================================

Metrics::AggregatedStats Metrics::ComputeEpisodeStats(const std::vector<EpisodeStats>& episodes) {
    if (episodes.empty()) {
        return AggregatedStats{};
    }

    AggregatedStats stats;
    stats.num_episodes = static_cast<int>(episodes.size());

    // Extract rewards and other data
    auto rewards = ExtractRewards(episodes);
    auto lengths = ExtractEpisodeLengths(episodes);
    auto successes = ExtractSuccesses(episodes);

    // Basic reward statistics
    stats.mean_reward = ComputeMeanReward(rewards);
    stats.std_reward = ComputeStdReward(rewards);
    auto reward_range = ComputeRewardRange(rewards);
    stats.min_reward = reward_range.first;
    stats.max_reward = reward_range.second;

    // Episode length and success rate
    stats.mean_episode_length = ComputeMeanEpisodeLength(lengths);
    stats.success_rate = ComputeSuccessRate(successes);

    // Custom metrics
    stats.custom_metrics = ComputeCustomMetrics(episodes);

    return stats;
}

// ============================================================================
// Reward Metrics
// ============================================================================

float Metrics::ComputeMeanReward(const std::vector<float>& rewards) {
    if (rewards.empty()) return 0.0f;
    return std::accumulate(rewards.begin(), rewards.end(), 0.0f) / rewards.size();
}

float Metrics::ComputeStdReward(const std::vector<float>& rewards) {
    if (rewards.size() <= 1) return 0.0f;
    
    float mean = ComputeMeanReward(rewards);
    float variance = 0.0f;
    
    for (float reward : rewards) {
        variance += (reward - mean) * (reward - mean);
    }
    
    return std::sqrt(variance / (rewards.size() - 1));
}

float Metrics::ComputeMedianReward(const std::vector<float>& rewards) {
    if (rewards.empty()) return 0.0f;
    
    std::vector<float> sorted_rewards = rewards;
    std::sort(sorted_rewards.begin(), sorted_rewards.end());
    
    size_t n = sorted_rewards.size();
    if (n % 2 == 0) {
        return (sorted_rewards[n/2 - 1] + sorted_rewards[n/2]) / 2.0f;
    } else {
        return sorted_rewards[n/2];
    }
}

std::pair<float, float> Metrics::ComputeRewardRange(const std::vector<float>& rewards) {
    if (rewards.empty()) return {0.0f, 0.0f};
    
    auto minmax = std::minmax_element(rewards.begin(), rewards.end());
    return {*minmax.first, *minmax.second};
}

// ============================================================================
// Episode Length and Success Metrics
// ============================================================================

float Metrics::ComputeMeanEpisodeLength(const std::vector<int64_t>& lengths) {
    if (lengths.empty()) return 0.0f;
    return static_cast<float>(std::accumulate(lengths.begin(), lengths.end(), 0LL)) / lengths.size();
}

float Metrics::ComputeSuccessRate(const std::vector<bool>& successes) {
    if (successes.empty()) return 0.0f;
    int success_count = std::count(successes.begin(), successes.end(), true);
    return static_cast<float>(success_count) / successes.size();
}

// ============================================================================
// Policy Evaluation Metrics
// ============================================================================

float Metrics::ComputePolicyEntropy(const torch::Tensor& action_probs) {
    if (action_probs.numel() == 0) return 0.0f;
    
    // Ensure probabilities are normalized and non-zero
    auto probs = torch::clamp(action_probs, 1e-8, 1.0);
    auto log_probs = torch::log(probs);
    auto entropy = -torch::sum(probs * log_probs, -1);
    
    return entropy.mean().item<float>();
}

float Metrics::ComputeValueEstimateAccuracy(const torch::Tensor& values, const torch::Tensor& returns) {
    if (values.numel() == 0 || returns.numel() == 0) return 0.0f;
    
    auto mse = torch::mse_loss(values, returns);
    return mse.item<float>();
}

float Metrics::ComputeExplainedVariance(const torch::Tensor& values, const torch::Tensor& returns) {
    if (values.numel() == 0 || returns.numel() == 0) return 0.0f;
    
    auto returns_var = torch::var(returns);
    auto residuals = returns - values;
    auto residuals_var = torch::var(residuals);
    
    if (returns_var.item<float>() == 0.0f) return 0.0f;
    
    return 1.0f - (residuals_var / returns_var).item<float>();
}

// ============================================================================
// Learning Progress Metrics
// ============================================================================

float Metrics::ComputeLearningCurveSlope(const std::vector<float>& rewards, int window_size) {
    if (rewards.size() < static_cast<size_t>(window_size) || window_size < 2) return 0.0f;
    
    // Take the last window_size rewards
    std::vector<float> window(rewards.end() - window_size, rewards.end());
    
    // Simple linear regression slope
    float n = static_cast<float>(window_size);
    float sum_x = n * (n - 1) / 2;  // 0 + 1 + ... + (n-1)
    float sum_y = std::accumulate(window.begin(), window.end(), 0.0f);
    float sum_xy = 0.0f;
    float sum_x2 = n * (n - 1) * (2 * n - 1) / 6;  // Sum of squares
    
    for (int i = 0; i < window_size; ++i) {
        sum_xy += i * window[i];
    }
    
    float denominator = n * sum_x2 - sum_x * sum_x;
    if (std::abs(denominator) < 1e-8) return 0.0f;
    
    return (n * sum_xy - sum_x * sum_y) / denominator;
}

float Metrics::ComputeStability(const std::vector<float>& rewards, int window_size) {
    if (rewards.size() < static_cast<size_t>(window_size)) return 0.0f;
    
    std::vector<float> window(rewards.end() - window_size, rewards.end());
    float mean = ComputeMeanReward(window);
    float variance = 0.0f;
    
    for (float reward : window) {
        variance += (reward - mean) * (reward - mean);
    }
    
    variance /= window_size;
    
    // Stability is inverse of coefficient of variation
    if (std::abs(mean) < 1e-8) return 0.0f;
    return 1.0f / (std::sqrt(variance) / std::abs(mean));
}

// ============================================================================
// Model Performance Metrics
// ============================================================================

std::unordered_map<std::string, float> Metrics::ComputeModelMetrics(
    const torch::Tensor& predictions, 
    const torch::Tensor& targets) {
    
    std::unordered_map<std::string, float> metrics;
    
    if (predictions.numel() == 0 || targets.numel() == 0) {
        return metrics;
    }
    
    // Mean Squared Error
    auto mse = torch::mse_loss(predictions, targets);
    metrics["mse"] = mse.item<float>();
    
    // Mean Absolute Error
    auto mae = torch::mean(torch::abs(predictions - targets));
    metrics["mae"] = mae.item<float>();
    
    // R-squared
    auto targets_mean = torch::mean(targets);
    auto ss_tot = torch::sum(torch::pow(targets - targets_mean, 2));
    auto ss_res = torch::sum(torch::pow(targets - predictions, 2));
    
    if (ss_tot.item<float>() > 1e-8) {
        metrics["r2"] = 1.0f - (ss_res / ss_tot).item<float>();
    } else {
        metrics["r2"] = 0.0f;
    }
    
    return metrics;
}

// ============================================================================
// Gradient and Weight Metrics
// ============================================================================

float Metrics::ComputeGradientNorm(const std::vector<torch::Tensor>& gradients) {
    if (gradients.empty()) return 0.0f;
    
    float total_norm = 0.0f;
    for (const auto& grad : gradients) {
        if (grad.defined()) {
            total_norm += torch::norm(grad).pow(2).item<float>();
        }
    }
    
    return std::sqrt(total_norm);
}

std::unordered_map<std::string, float> Metrics::ComputeWeightStats(
    const std::vector<torch::Tensor>& weights) {
    
    std::unordered_map<std::string, float> stats;
    
    if (weights.empty()) return stats;
    
    std::vector<float> all_weights;
    
    // Flatten all weights
    for (const auto& weight : weights) {
        if (weight.defined() && weight.numel() > 0) {
            auto flattened = weight.flatten();
            for (int64_t i = 0; i < flattened.numel(); ++i) {
                all_weights.push_back(flattened[i].item<float>());
            }
        }
    }
    
    if (all_weights.empty()) return stats;
    
    // Compute statistics
    float mean = std::accumulate(all_weights.begin(), all_weights.end(), 0.0f) / all_weights.size();
    stats["mean"] = mean;
    
    float variance = 0.0f;
    for (float w : all_weights) {
        variance += (w - mean) * (w - mean);
    }
    variance /= all_weights.size();
    stats["std"] = std::sqrt(variance);
    
    auto minmax = std::minmax_element(all_weights.begin(), all_weights.end());
    stats["min"] = *minmax.first;
    stats["max"] = *minmax.second;
    
    // Count near-zero weights
    int zero_count = std::count_if(all_weights.begin(), all_weights.end(), 
                                  [](float w) { return std::abs(w) < 1e-6; });
    stats["sparsity"] = static_cast<float>(zero_count) / all_weights.size();
    
    return stats;
}

// ============================================================================
// Custom Metrics
// ============================================================================

void Metrics::RegisterCustomMetric(const std::string& name, const CustomMetricFunction& metric_fn) {
    custom_metrics_[name] = metric_fn;
}

std::unordered_map<std::string, float> Metrics::ComputeCustomMetrics(
    const std::vector<EpisodeStats>& episodes) {
    
    std::unordered_map<std::string, float> results;
    
    for (const auto& [name, metric_fn] : custom_metrics_) {
        try {
            results[name] = metric_fn(episodes);
        } catch (const std::exception& e) {
            std::cerr << "Error computing custom metric '" << name << "': " << e.what() << std::endl;
            results[name] = 0.0f;
        }
    }
    
    return results;
}

// ============================================================================
// Helper Functions
// ============================================================================

std::vector<float> Metrics::ExtractRewards(const std::vector<EpisodeStats>& episodes) {
    std::vector<float> rewards;
    rewards.reserve(episodes.size());
    
    for (const auto& episode : episodes) {
        rewards.push_back(episode.total_reward);
    }
    
    return rewards;
}

std::vector<int64_t> Metrics::ExtractEpisodeLengths(const std::vector<EpisodeStats>& episodes) {
    std::vector<int64_t> lengths;
    lengths.reserve(episodes.size());
    
    for (const auto& episode : episodes) {
        lengths.push_back(episode.total_steps);
    }
    
    return lengths;
}

std::vector<bool> Metrics::ExtractSuccesses(const std::vector<EpisodeStats>& episodes) {
    std::vector<bool> successes;
    successes.reserve(episodes.size());
    
    for (const auto& episode : episodes) {
        successes.push_back(episode.success);
    }
    
    return successes;
}

} // namespace training