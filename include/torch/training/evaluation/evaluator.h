#pragma once

#include "metrics.h"
#include <torch/torch.h>
#include <memory>
#include <vector>
#include <functional>

// Forward declarations
class AbstractEnv;

namespace networks {
    class NetworkBase;
}

namespace training {

/**
 * Evaluator for reinforcement learning agents
 * Handles evaluation of trained policies on environments
 */
class Evaluator {
public:
    struct Config {
        int num_eval_episodes = 10;
        int max_episode_steps = 1000;
        bool render_evaluation = false;
        bool deterministic_policy = true;
        torch::Device device = torch::kCPU;
        
        // Recording options
        bool record_episodes = false;
        std::string recording_dir = "./recordings";
        
        // Evaluation metrics
        std::vector<std::string> metrics_to_compute = {"reward", "episode_length", "success_rate"};
        bool compute_policy_metrics = true;
        bool compute_value_metrics = false;
    };

    Evaluator(const Config& config);
    ~Evaluator() = default;

    // Main evaluation interface
    Metrics::AggregatedStats EvaluatePolicy(
        std::shared_ptr<networks::NetworkBase> policy,
        std::vector<std::unique_ptr<AbstractEnv>>& eval_envs);
    
    // Single environment evaluation
    Metrics::EpisodeStats EvaluateEpisode(
        std::shared_ptr<networks::NetworkBase> policy,
        AbstractEnv& env);
    
    // Batch evaluation (multiple environments in parallel)
    std::vector<Metrics::EpisodeStats> EvaluateBatch(
        std::shared_ptr<networks::NetworkBase> policy,
        std::vector<std::unique_ptr<AbstractEnv>>& eval_envs);

    // Policy-specific evaluation
    using ActionSelector = std::function<torch::Tensor(const torch::Tensor&)>;
    Metrics::EpisodeStats EvaluateWithActionSelector(
        const ActionSelector& action_selector,
        AbstractEnv& env);

    // Advanced evaluation methods
    std::vector<float> EvaluateReturns(
        std::shared_ptr<networks::NetworkBase> policy,
        std::vector<std::unique_ptr<AbstractEnv>>& eval_envs,
        float gamma = 0.99f);
    
    float EvaluateValueFunction(
        std::shared_ptr<networks::NetworkBase> value_function,
        const torch::Tensor& states,
        const torch::Tensor& returns);

    // Configuration
    void SetNumEpisodes(int num_episodes) { config_.num_eval_episodes = num_episodes; }
    void SetMaxEpisodeSteps(int max_steps) { config_.max_episode_steps = max_steps; }
    void SetDeterministic(bool deterministic) { config_.deterministic_policy = deterministic; }
    void SetRenderEvaluation(bool render) { config_.render_evaluation = render; }
    void SetDevice(const torch::Device& device) { config_.device = device; }

    // Episode recording
    void EnableRecording(const std::string& recording_dir);
    void DisableRecording();

    // Custom metrics
    using EpisodeHook = std::function<void(const Metrics::EpisodeStats&)>;
    void SetEpisodeHook(const EpisodeHook& hook) { episode_hook_ = hook; }

    // Statistics
    const std::vector<Metrics::EpisodeStats>& GetLastEvaluationEpisodes() const { 
        return last_evaluation_episodes_; 
    }
    const Metrics::AggregatedStats& GetLastAggregatedStats() const { 
        return last_aggregated_stats_; 
    }

private:
    Config config_;
    EpisodeHook episode_hook_;
    
    // Last evaluation results
    std::vector<Metrics::EpisodeStats> last_evaluation_episodes_;
    Metrics::AggregatedStats last_aggregated_stats_;
    
    // Helper methods
    torch::Tensor SelectAction(std::shared_ptr<networks::NetworkBase> policy,
                              const torch::Tensor& observation);
    torch::Tensor SelectDeterministicAction(std::shared_ptr<networks::NetworkBase> policy,
                                           const torch::Tensor& observation);
    torch::Tensor SelectStochasticAction(std::shared_ptr<networks::NetworkBase> policy,
                                        const torch::Tensor& observation);
    
    void RecordEpisode(const Metrics::EpisodeStats& episode_stats);
    void LogEvaluationResults(const Metrics::AggregatedStats& stats) const;
    
    // Environment utilities
    torch::Tensor GetObservation(AbstractEnv& env);
    void ResetEnvironment(AbstractEnv& env);
    bool StepEnvironment(AbstractEnv& env, const torch::Tensor& action,
                        float& reward, bool& done);
};

} // namespace training
