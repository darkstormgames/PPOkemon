
#include "torch/training/evaluation/evaluator.h"
#include "torch/envs/env_abstract.h"
#include "torch/networks/network_base.h"
#include <iostream>
#include <filesystem>
#include <fstream>

namespace training {

Evaluator::Evaluator(const Config& config) : config_(config) {
    if (config_.record_episodes) {
        std::filesystem::create_directories(config_.recording_dir);
    }
}

// ============================================================================
// Main Evaluation Interface
// ============================================================================

Metrics::AggregatedStats Evaluator::EvaluatePolicy(
    std::shared_ptr<networks::NetworkBase> policy,
    std::vector<std::unique_ptr<AbstractEnv>>& eval_envs) {
    
    if (!policy) {
        throw std::runtime_error("Policy network is null");
    }
    
    if (eval_envs.empty()) {
        throw std::runtime_error("No evaluation environments provided");
    }
    
    // Set policy to evaluation mode
    policy->eval();
    
    std::vector<Metrics::EpisodeStats> episode_stats;
    episode_stats.reserve(config_.num_eval_episodes);
    
    // Distribute episodes across environments
    int episodes_per_env = config_.num_eval_episodes / static_cast<int>(eval_envs.size());
    int remaining_episodes = config_.num_eval_episodes % static_cast<int>(eval_envs.size());
    
    for (size_t env_idx = 0; env_idx < eval_envs.size() && static_cast<int>(episode_stats.size()) < config_.num_eval_episodes; ++env_idx) {
        int episodes_for_this_env = episodes_per_env + (static_cast<int>(env_idx) < remaining_episodes ? 1 : 0);
        
        for (int ep = 0; ep < episodes_for_this_env; ++ep) {
            auto episode_stat = EvaluateEpisode(policy, *eval_envs[env_idx]);
            episode_stats.push_back(episode_stat);
            
            // Call episode hook if registered
            if (episode_hook_) {
                episode_hook_(episode_stat);
            }
            
            // Record episode if enabled
            if (config_.record_episodes) {
                RecordEpisode(episode_stat);
            }
        }
    }
    
    // Store results
    last_evaluation_episodes_ = episode_stats;
    last_aggregated_stats_ = Metrics::ComputeEpisodeStats(episode_stats);
    
    LogEvaluationResults(last_aggregated_stats_);
    
    return last_aggregated_stats_;
}

Metrics::EpisodeStats Evaluator::EvaluateEpisode(
    std::shared_ptr<networks::NetworkBase> policy,
    AbstractEnv& env) {
    
    Metrics::EpisodeStats stats;
    ResetEnvironment(env);
    
    auto observation = GetObservation(env);
    
    for (int64_t step = 0; step < config_.max_episode_steps; ++step) {
        // Select action
        auto action = SelectAction(policy, observation);
        
        // Step environment
        float reward;
        bool done;
        bool success = StepEnvironment(env, action, reward, done);
        
        // Update statistics
        stats.total_reward += reward;
        stats.total_steps = step + 1;
        
        if (done) {
            stats.success = success;
            break;
        }
        
        // Get next observation
        observation = GetObservation(env);
        
        // Optional: render evaluation
        if (config_.render_evaluation) {
            // env.Render(); // Uncomment when rendering is implemented
        }
    }
    
    stats.episode_length = static_cast<float>(stats.total_steps);
    
    return stats;
}

std::vector<Metrics::EpisodeStats> Evaluator::EvaluateBatch(
    std::shared_ptr<networks::NetworkBase> policy,
    std::vector<std::unique_ptr<AbstractEnv>>& eval_envs) {
    
    std::vector<Metrics::EpisodeStats> all_stats;
    
    for (auto& env : eval_envs) {
        auto stats = EvaluateEpisode(policy, *env);
        all_stats.push_back(stats);
    }
    
    return all_stats;
}

Metrics::EpisodeStats Evaluator::EvaluateWithActionSelector(
    const ActionSelector& action_selector,
    AbstractEnv& env) {
    
    Metrics::EpisodeStats stats;
    ResetEnvironment(env);
    
    auto observation = GetObservation(env);
    
    for (int64_t step = 0; step < config_.max_episode_steps; ++step) {
        // Select action using custom selector
        auto action = action_selector(observation);
        
        // Step environment
        float reward;
        bool done;
        bool success = StepEnvironment(env, action, reward, done);
        
        // Update statistics
        stats.total_reward += reward;
        stats.total_steps = step + 1;
        
        if (done) {
            stats.success = success;
            break;
        }
        
        // Get next observation
        observation = GetObservation(env);
    }
    
    stats.episode_length = static_cast<float>(stats.total_steps);
    
    return stats;
}

// ============================================================================
// Advanced Evaluation Methods
// ============================================================================

std::vector<float> Evaluator::EvaluateReturns(
    std::shared_ptr<networks::NetworkBase> policy,
    std::vector<std::unique_ptr<AbstractEnv>>& eval_envs,
    float /*gamma*/) {
    
    auto episode_stats = EvaluateBatch(policy, eval_envs);
    std::vector<float> returns;
    returns.reserve(episode_stats.size());
    
    for (const auto& stats : episode_stats) {
        // For now, just use undiscounted return (total_reward)
        // In a more sophisticated implementation, we would track step-by-step rewards
        // and compute discounted returns
        returns.push_back(stats.total_reward);
    }
    
    return returns;
}

float Evaluator::EvaluateValueFunction(
    std::shared_ptr<networks::NetworkBase> value_function,
    const torch::Tensor& states,
    const torch::Tensor& returns) {
    
    if (!value_function) {
        return 0.0f;
    }
    
    value_function->eval();
    
    torch::NoGradGuard no_grad;
    
    // Get value predictions
    auto values = value_function->forward(states);
    
    // Compute value function accuracy
    return Metrics::ComputeValueEstimateAccuracy(values, returns);
}

// ============================================================================
// Episode Recording
// ============================================================================

void Evaluator::EnableRecording(const std::string& recording_dir) {
    config_.record_episodes = true;
    config_.recording_dir = recording_dir;
    std::filesystem::create_directories(recording_dir);
}

void Evaluator::DisableRecording() {
    config_.record_episodes = false;
}

// ============================================================================
// Action Selection Methods
// ============================================================================

torch::Tensor Evaluator::SelectAction(std::shared_ptr<networks::NetworkBase> policy,
                                     const torch::Tensor& observation) {
    if (config_.deterministic_policy) {
        return SelectDeterministicAction(policy, observation);
    } else {
        return SelectStochasticAction(policy, observation);
    }
}

torch::Tensor Evaluator::SelectDeterministicAction(std::shared_ptr<networks::NetworkBase> policy,
                                                  const torch::Tensor& observation) {
    torch::NoGradGuard no_grad;
    policy->to(config_.device);
    
    auto obs_batch = observation.unsqueeze(0).to(config_.device);
    auto output = policy->forward(obs_batch);
    
    // For deterministic evaluation, select the action with highest probability
    if (output.dim() == 2) {
        // Discrete actions - select argmax
        return torch::argmax(output, /*dim=*/1).squeeze(0).cpu();
    } else {
        // Continuous actions - use mean (assuming network outputs mean and std)
        return output.squeeze(0).cpu();
    }
}

torch::Tensor Evaluator::SelectStochasticAction(std::shared_ptr<networks::NetworkBase> policy,
                                               const torch::Tensor& observation) {
    torch::NoGradGuard no_grad;
    policy->to(config_.device);
    
    auto obs_batch = observation.unsqueeze(0).to(config_.device);
    auto output = policy->forward(obs_batch);
    
    if (output.dim() == 2) {
        // Discrete actions - sample from distribution
        auto probs = torch::softmax(output, /*dim=*/1);
        auto action = torch::multinomial(probs, /*num_samples=*/1);
        return action.squeeze().cpu();
    } else {
        // Continuous actions - add some noise for exploration
        auto noise = torch::randn_like(output) * 0.1f;
        return (output + noise).squeeze(0).cpu();
    }
}

// ============================================================================
// Helper Methods
// ============================================================================

torch::Tensor Evaluator::GetObservation(AbstractEnv& /*env*/) {
    // This is a placeholder - the actual implementation depends on the environment interface
    // For now, create a dummy observation
    return torch::randn({4}); // Placeholder observation
}

void Evaluator::ResetEnvironment(AbstractEnv& /*env*/) {
    // This is a placeholder - the actual implementation depends on the environment interface
    // env.Reset();
}

bool Evaluator::StepEnvironment(AbstractEnv& /*env*/, const torch::Tensor& /*action*/,
                               float& reward, bool& done) {
    // This is a placeholder - the actual implementation depends on the environment interface
    // For now, simulate a simple environment step
    reward = 1.0f; // Dummy reward
    done = false;  // Dummy done flag
    
    // Randomly terminate episodes for testing
    static int step_count = 0;
    step_count++;
    if (step_count % 100 == 0) {
        done = true;
        step_count = 0;
    }
    
    return done; // Return success status (same as done for now)
}

void Evaluator::RecordEpisode(const Metrics::EpisodeStats& episode_stats) {
    if (!config_.record_episodes) return;
    
    static int episode_id = 0;
    std::string filename = config_.recording_dir + "/episode_" + std::to_string(episode_id++) + ".json";
    
    std::ofstream file(filename);
    if (file.is_open()) {
        // Simple JSON-like format
        file << "{\n";
        file << "  \"total_reward\": " << episode_stats.total_reward << ",\n";
        file << "  \"total_steps\": " << episode_stats.total_steps << ",\n";
        file << "  \"episode_length\": " << episode_stats.episode_length << ",\n";
        file << "  \"success\": " << (episode_stats.success ? "true" : "false") << "\n";
        file << "}\n";
        file.close();
    }
}

void Evaluator::LogEvaluationResults(const Metrics::AggregatedStats& stats) const {
    if (config_.num_eval_episodes > 0) {
        std::cout << "\n=== Evaluation Results ===\n";
        std::cout << "Episodes: " << stats.num_episodes << "\n";
        std::cout << "Mean Reward: " << stats.mean_reward << " ± " << stats.std_reward << "\n";
        std::cout << "Reward Range: [" << stats.min_reward << ", " << stats.max_reward << "]\n";
        std::cout << "Mean Episode Length: " << stats.mean_episode_length << "\n";
        std::cout << "Success Rate: " << (stats.success_rate * 100.0f) << "%\n";
        
        if (!stats.custom_metrics.empty()) {
            std::cout << "Custom Metrics:\n";
            for (const auto& [name, value] : stats.custom_metrics) {
                std::cout << "  " << name << ": " << value << "\n";
            }
        }
        std::cout << "==========================\n\n";
    }
}

} // namespace training