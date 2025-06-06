#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

// Forward declarations
class AbstractEnv;

namespace networks {
    class NetworkBase;
}

namespace training {

/**
 * Evaluation metrics for reinforcement learning
 */
class Metrics {
public:
    struct EpisodeStats {
        float total_reward = 0.0f;
        int64_t total_steps = 0;
        float episode_length = 0.0f;
        bool success = false;
        std::unordered_map<std::string, float> custom_metrics;
    };

    struct AggregatedStats {
        float mean_reward = 0.0f;
        float std_reward = 0.0f;
        float min_reward = 0.0f;
        float max_reward = 0.0f;
        float mean_episode_length = 0.0f;
        float success_rate = 0.0f;
        int num_episodes = 0;
        std::unordered_map<std::string, float> custom_metrics;
    };

    // Compute basic statistics from episode data
    static AggregatedStats ComputeEpisodeStats(const std::vector<EpisodeStats>& episodes);
    
    // Reward-based metrics
    static float ComputeMeanReward(const std::vector<float>& rewards);
    static float ComputeStdReward(const std::vector<float>& rewards);
    static float ComputeMedianReward(const std::vector<float>& rewards);
    static std::pair<float, float> ComputeRewardRange(const std::vector<float>& rewards);
    
    // Episode length metrics
    static float ComputeMeanEpisodeLength(const std::vector<int64_t>& lengths);
    static float ComputeSuccessRate(const std::vector<bool>& successes);
    
    // Policy evaluation metrics
    static float ComputePolicyEntropy(const torch::Tensor& action_probs);
    static float ComputeValueEstimateAccuracy(const torch::Tensor& values, 
                                             const torch::Tensor& returns);
    static float ComputeExplainedVariance(const torch::Tensor& values, 
                                         const torch::Tensor& returns);
    
    // Learning progress metrics
    static float ComputeLearningCurveSlope(const std::vector<float>& rewards, 
                                          int window_size = 100);
    static float ComputeStability(const std::vector<float>& rewards, 
                                 int window_size = 100);
    
    // Model performance metrics
    static std::unordered_map<std::string, float> ComputeModelMetrics(
        const torch::Tensor& predictions, 
        const torch::Tensor& targets);
    
    // Gradient and training metrics
    static float ComputeGradientNorm(const std::vector<torch::Tensor>& gradients);
    static std::unordered_map<std::string, float> ComputeWeightStats(
        const std::vector<torch::Tensor>& weights);
    
    // Custom metric registration
    using CustomMetricFunction = std::function<float(const std::vector<EpisodeStats>&)>;
    static void RegisterCustomMetric(const std::string& name, 
                                   const CustomMetricFunction& metric_fn);
    static std::unordered_map<std::string, float> ComputeCustomMetrics(
        const std::vector<EpisodeStats>& episodes);

private:
    static std::unordered_map<std::string, CustomMetricFunction> custom_metrics_;
    
    // Helper functions
    static std::vector<float> ExtractRewards(const std::vector<EpisodeStats>& episodes);
    static std::vector<int64_t> ExtractEpisodeLengths(const std::vector<EpisodeStats>& episodes);
    static std::vector<bool> ExtractSuccesses(const std::vector<EpisodeStats>& episodes);
};

} // namespace training
