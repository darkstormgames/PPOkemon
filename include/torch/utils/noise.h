#pragma once

#include <torch/torch.h>
#include <memory>
#include <random>

namespace utils {

/**
 * Base class for noise generators
 */
class NoiseGenerator {
public:
    virtual ~NoiseGenerator() = default;
    virtual torch::Tensor Sample(const torch::IntArrayRef& shape, const torch::Device& device = torch::kCPU) = 0;
    virtual void Reset() {}
    virtual std::unique_ptr<NoiseGenerator> Clone() const = 0;
};

/**
 * Gaussian (normal) noise generator
 */
class GaussianNoise : public NoiseGenerator {
public:
    explicit GaussianNoise(float mean = 0.0f, float std = 1.0f, unsigned int seed = 0);
    
    torch::Tensor Sample(const torch::IntArrayRef& shape, const torch::Device& device = torch::kCPU) override;
    void Reset() override {}
    std::unique_ptr<NoiseGenerator> Clone() const override;
    
    void SetMean(float mean) { mean_ = mean; }
    void SetStd(float std) { std_ = std; }
    float GetMean() const { return mean_; }
    float GetStd() const { return std_; }

private:
    float mean_;
    float std_;
    unsigned int seed_;
};

/**
 * Ornstein-Uhlenbeck noise for continuous control
 */
class OrnsteinUhlenbeckNoise : public NoiseGenerator {
public:
    explicit OrnsteinUhlenbeckNoise(float theta = 0.15f, 
                                   float sigma = 0.2f, 
                                   float mu = 0.0f,
                                   float dt = 1e-2f,
                                   unsigned int seed = 0);
    
    torch::Tensor Sample(const torch::IntArrayRef& shape, const torch::Device& device = torch::kCPU) override;
    void Reset() override;
    std::unique_ptr<NoiseGenerator> Clone() const override;
    
    void SetParameters(float theta, float sigma, float mu, float dt);
    
private:
    float theta_;   // Mean reversion rate
    float sigma_;   // Volatility
    float mu_;      // Long-term mean
    float dt_;      // Time step
    
    torch::Tensor state_;
    std::mt19937 generator_;
    std::normal_distribution<float> distribution_;
    bool state_initialized_;
};

/**
 * Parameter noise for neural network weights
 */
class ParameterNoise : public NoiseGenerator {
public:
    explicit ParameterNoise(float std = 0.1f, 
                           float std_decay = 0.99f,
                           float min_std = 0.01f,
                           unsigned int seed = 0);
    
    torch::Tensor Sample(const torch::IntArrayRef& shape, const torch::Device& device = torch::kCPU) override;
    void Reset() override {}
    std::unique_ptr<NoiseGenerator> Clone() const override;
    
    // Apply noise to network parameters
    void ApplyToNetwork(torch::nn::Module& network, const std::string& prefix = "");
    void RemoveFromNetwork(torch::nn::Module& network, const std::string& prefix = "");
    
    // Adaptive noise scaling
    void AdaptNoise(float reward_diff);
    void DecayNoise() { current_std_ *= std_decay_; current_std_ = std::max(current_std_, min_std_); }
    
    float GetCurrentStd() const { return current_std_; }

private:
    float base_std_;
    float current_std_;
    float std_decay_;
    float min_std_;
    std::mt19937 generator_;
    std::normal_distribution<float> distribution_;
    
    // Store original weights for restoration
    std::unordered_map<std::string, torch::Tensor> original_weights_;
};

/**
 * Epsilon-greedy noise for discrete actions
 */
class EpsilonGreedyNoise {
public:
    explicit EpsilonGreedyNoise(float epsilon = 0.1f, 
                               float epsilon_decay = 0.995f,
                               float min_epsilon = 0.01f,
                               unsigned int seed = 0);
    
    // Returns true if should take random action
    bool ShouldExplore() const;
    
    // Sample random action from action space
    int SampleRandomAction(int num_actions) const;
    
    // Decay epsilon
    void DecayEpsilon();
    void Reset() { current_epsilon_ = base_epsilon_; }
    
    // Getters/Setters
    void SetEpsilon(float epsilon) { current_epsilon_ = epsilon; }
    float GetEpsilon() const { return current_epsilon_; }

private:
    float base_epsilon_;
    float current_epsilon_;
    float epsilon_decay_;
    float min_epsilon_;
    mutable std::mt19937 generator_;
    mutable std::uniform_real_distribution<float> uniform_dist_;
    mutable std::uniform_int_distribution<int> action_dist_;
};

/**
 * Scheduled noise that changes parameters over time
 */
class ScheduledNoise : public NoiseGenerator {
public:
    explicit ScheduledNoise(std::unique_ptr<NoiseGenerator> base_noise,
                           float initial_scale = 1.0f,
                           float final_scale = 0.1f,
                           int64_t schedule_steps = 1000000);
    
    torch::Tensor Sample(const torch::IntArrayRef& shape, const torch::Device& device = torch::kCPU) override;
    void Reset() override { base_noise_->Reset(); current_step_ = 0; }
    std::unique_ptr<NoiseGenerator> Clone() const override;
    
    void Step() { current_step_++; }
    void SetStep(int64_t step) { current_step_ = step; }
    float GetCurrentScale() const;

private:
    std::unique_ptr<NoiseGenerator> base_noise_;
    float initial_scale_;
    float final_scale_;
    int64_t schedule_steps_;
    int64_t current_step_;
};

/**
 * Utility class for applying different types of noise
 */
class NoiseManager {
public:
    NoiseManager() = default;
    
    // Register noise generators
    void RegisterNoise(const std::string& name, std::unique_ptr<NoiseGenerator> noise);
    
    // Apply noise to tensors
    torch::Tensor ApplyNoise(const std::string& name, const torch::Tensor& tensor);
    
    // Apply noise to network parameters
    void ApplyParameterNoise(const std::string& name, torch::nn::Module& network);
    
    // Noise scheduling and adaptation
    void StepSchedules();
    void ResetAll();
    
    // Get noise generator
    NoiseGenerator* GetNoise(const std::string& name);

private:
    std::unordered_map<std::string, std::unique_ptr<NoiseGenerator>> noise_generators_;
};

} // namespace utils
