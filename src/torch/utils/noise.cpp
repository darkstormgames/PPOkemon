#include "torch/utils/noise.h"
#include <stdexcept>
#include <cmath>

namespace utils {

// GaussianNoise implementation
GaussianNoise::GaussianNoise(float mean, float std, unsigned int seed)
    : mean_(mean), std_(std), seed_(seed) {
    if (std <= 0.0f) {
        throw std::invalid_argument("Standard deviation must be positive");
    }
}

torch::Tensor GaussianNoise::Sample(const torch::IntArrayRef& shape, const torch::Device& device) {
    torch::Tensor noise = torch::randn(shape, torch::TensorOptions().device(device));
    return noise * std_ + mean_;
}

std::unique_ptr<NoiseGenerator> GaussianNoise::Clone() const {
    return std::make_unique<GaussianNoise>(mean_, std_, seed_);
}

// OrnsteinUhlenbeckNoise implementation
OrnsteinUhlenbeckNoise::OrnsteinUhlenbeckNoise(float theta, float sigma, float mu, float dt, unsigned int seed)
    : theta_(theta), sigma_(sigma), mu_(mu), dt_(dt), generator_(seed), 
      distribution_(0.0f, 1.0f), state_initialized_(false) {}

torch::Tensor OrnsteinUhlenbeckNoise::Sample(const torch::IntArrayRef& shape, const torch::Device& device) {
    if (!state_initialized_ || !state_.defined() || !state_.sizes().equals(shape)) {
        state_ = torch::full(shape, mu_, torch::TensorOptions().device(device));
        state_initialized_ = true;
    }
    
    // Move state to correct device if needed
    if (state_.device() != device) {
        state_ = state_.to(device);
    }
    
    // OU process: dx = theta * (mu - x) * dt + sigma * sqrt(dt) * dW
    torch::Tensor dw = torch::zeros(shape, torch::TensorOptions().device(device));
    
    // Fill with random samples
    auto dw_accessor = dw.accessor<float, 1>();
    for (int64_t i = 0; i < dw.numel(); ++i) {
        dw_accessor[i] = distribution_(generator_);
    }
    
    torch::Tensor dx = theta_ * (mu_ - state_) * dt_ + sigma_ * std::sqrt(dt_) * dw;
    state_ = state_ + dx;
    
    return state_.clone();
}

void OrnsteinUhlenbeckNoise::Reset() {
    state_initialized_ = false;
}

std::unique_ptr<NoiseGenerator> OrnsteinUhlenbeckNoise::Clone() const {
    return std::make_unique<OrnsteinUhlenbeckNoise>(theta_, sigma_, mu_, dt_);
}

void OrnsteinUhlenbeckNoise::SetParameters(float theta, float sigma, float mu, float dt) {
    theta_ = theta;
    sigma_ = sigma;
    mu_ = mu;
    dt_ = dt;
}

// ParameterNoise implementation
ParameterNoise::ParameterNoise(float std, float std_decay, float min_std, unsigned int seed)
    : base_std_(std), current_std_(std), std_decay_(std_decay), min_std_(min_std),
      generator_(seed), distribution_(0.0f, 1.0f) {}

torch::Tensor ParameterNoise::Sample(const torch::IntArrayRef& shape, const torch::Device& device) {
    torch::Tensor noise = torch::randn(shape, torch::TensorOptions().device(device));
    return noise * current_std_;
}

std::unique_ptr<NoiseGenerator> ParameterNoise::Clone() const {
    return std::make_unique<ParameterNoise>(base_std_, std_decay_, min_std_);
}

void ParameterNoise::ApplyToNetwork(torch::nn::Module& network, const std::string& prefix) {
    torch::NoGradGuard no_grad;
    
    for (auto& param : network.named_parameters()) {
        const std::string& name = param.key();
        torch::Tensor& weight = param.value();
        
        std::string full_name = prefix.empty() ? name : prefix + "." + name;
        
        // Store original weight if not already stored
        if (original_weights_.find(full_name) == original_weights_.end()) {
            original_weights_[full_name] = weight.clone();
        }
        
        // Apply noise
        torch::Tensor noise = Sample(weight.sizes(), weight.device());
        weight.add_(noise);
    }
}

void ParameterNoise::RemoveFromNetwork(torch::nn::Module& network, const std::string& prefix) {
    torch::NoGradGuard no_grad;
    
    for (auto& param : network.named_parameters()) {
        const std::string& name = param.key();
        torch::Tensor& weight = param.value();
        
        std::string full_name = prefix.empty() ? name : prefix + "." + name;
        
        // Restore original weight
        auto it = original_weights_.find(full_name);
        if (it != original_weights_.end()) {
            weight.copy_(it->second);
        }
    }
    
    // Clear stored weights
    original_weights_.clear();
}

void ParameterNoise::AdaptNoise(float reward_diff) {
    // Adaptive noise scaling based on performance
    if (reward_diff > 0) {
        current_std_ *= 1.01f; // Increase noise if performance improved
    } else {
        current_std_ *= 0.99f; // Decrease noise if performance degraded
    }
    current_std_ = std::max(current_std_, min_std_);
}

// EpsilonGreedyNoise implementation
EpsilonGreedyNoise::EpsilonGreedyNoise(float epsilon, float epsilon_decay, float min_epsilon, unsigned int seed)
    : base_epsilon_(epsilon), current_epsilon_(epsilon), epsilon_decay_(epsilon_decay), 
      min_epsilon_(min_epsilon), generator_(seed), uniform_dist_(0.0f, 1.0f) {}

bool EpsilonGreedyNoise::ShouldExplore() const {
    return uniform_dist_(generator_) < current_epsilon_;
}

int EpsilonGreedyNoise::SampleRandomAction(int num_actions) const {
    std::uniform_int_distribution<int> action_dist(0, num_actions - 1);
    return action_dist(generator_);
}

void EpsilonGreedyNoise::DecayEpsilon() {
    current_epsilon_ = std::max(current_epsilon_ * epsilon_decay_, min_epsilon_);
}

// ScheduledNoise implementation
ScheduledNoise::ScheduledNoise(std::unique_ptr<NoiseGenerator> base_noise,
                               float initial_scale, float final_scale, int64_t schedule_steps)
    : base_noise_(std::move(base_noise)), initial_scale_(initial_scale), 
      final_scale_(final_scale), schedule_steps_(schedule_steps), current_step_(0) {}

torch::Tensor ScheduledNoise::Sample(const torch::IntArrayRef& shape, const torch::Device& device) {
    torch::Tensor base_sample = base_noise_->Sample(shape, device);
    float current_scale = GetCurrentScale();
    return base_sample * current_scale;
}

std::unique_ptr<NoiseGenerator> ScheduledNoise::Clone() const {
    return std::make_unique<ScheduledNoise>(base_noise_->Clone(), initial_scale_, 
                                           final_scale_, schedule_steps_);
}

float ScheduledNoise::GetCurrentScale() const {
    if (current_step_ >= schedule_steps_) {
        return final_scale_;
    }
    
    float progress = static_cast<float>(current_step_) / static_cast<float>(schedule_steps_);
    return initial_scale_ + (final_scale_ - initial_scale_) * progress;
}

// NoiseManager implementation
void NoiseManager::RegisterNoise(const std::string& name, std::unique_ptr<NoiseGenerator> noise) {
    noise_generators_[name] = std::move(noise);
}

torch::Tensor NoiseManager::ApplyNoise(const std::string& name, const torch::Tensor& tensor) {
    auto it = noise_generators_.find(name);
    if (it == noise_generators_.end()) {
        throw std::runtime_error("Noise generator '" + name + "' not found");
    }
    
    torch::Tensor noise = it->second->Sample(tensor.sizes(), tensor.device());
    return tensor + noise;
}

void NoiseManager::ApplyParameterNoise(const std::string& name, torch::nn::Module& network) {
    auto it = noise_generators_.find(name);
    if (it == noise_generators_.end()) {
        throw std::runtime_error("Noise generator '" + name + "' not found");
    }
    
    // Try to cast to ParameterNoise
    ParameterNoise* param_noise = dynamic_cast<ParameterNoise*>(it->second.get());
    if (param_noise) {
        param_noise->ApplyToNetwork(network);
    } else {
        throw std::runtime_error("Noise generator '" + name + "' is not a ParameterNoise");
    }
}

void NoiseManager::StepSchedules() {
    for (auto& [name, noise] : noise_generators_) {
        ScheduledNoise* scheduled = dynamic_cast<ScheduledNoise*>(noise.get());
        if (scheduled) {
            scheduled->Step();
        }
    }
}

void NoiseManager::ResetAll() {
    for (auto& [name, noise] : noise_generators_) {
        noise->Reset();
    }
}

NoiseGenerator* NoiseManager::GetNoise(const std::string& name) {
    auto it = noise_generators_.find(name);
    return (it != noise_generators_.end()) ? it->second.get() : nullptr;
}

} // namespace utils
