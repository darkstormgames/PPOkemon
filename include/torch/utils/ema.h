#pragma once

#include <torch/torch.h>
#include <unordered_map>
#include <string>

namespace utils {

/**
 * Exponential Moving Average (EMA) utility for model parameters
 * Useful for stable training and model averaging in DRL
 */
class ExponentialMovingAverage {
public:
    explicit ExponentialMovingAverage(float decay = 0.999f, const torch::Device& device = torch::kCPU);
    ~ExponentialMovingAverage();
    
    // Register parameters from a model
    void RegisterModel(const torch::nn::Module& model, const std::string& prefix = "");
    void RegisterParameter(const std::string& name, const torch::Tensor& param);
    
    // Update EMA parameters
    void Update(const torch::nn::Module& model, const std::string& prefix = "");
    void UpdateParameter(const std::string& name, const torch::Tensor& param);
    
    // Apply EMA parameters to model (for evaluation)
    void ApplyToModel(torch::nn::Module& model, const std::string& prefix = "") const;
    
    // Get EMA parameters
    torch::Tensor GetParameter(const std::string& name) const;
    bool HasParameter(const std::string& name) const;
    
    // Configuration
    void SetDecay(float decay) { decay_ = decay; }
    float GetDecay() const { return decay_; }
    
    // State management
    void Save(const std::string& path) const;
    void Load(const std::string& path);
    void Reset();
    
    // Statistics
    size_t GetParameterCount() const { return ema_params_.size(); }

private:
    float decay_;
    torch::Device device_;
    std::unordered_map<std::string, torch::Tensor> ema_params_;
    
    // Helper methods
    std::string GetParameterName(const std::string& prefix, const std::string& param_name) const;
    void CollectModelParameters(const torch::nn::Module& model, const std::string& prefix,
                               std::unordered_map<std::string, torch::Tensor>& params) const;
};

} // namespace utils
