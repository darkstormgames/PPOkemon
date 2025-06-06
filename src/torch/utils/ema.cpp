#include "torch/utils/ema.h"
#include <torch/torch.h>
#include <stdexcept>
#include <fstream>

namespace utils {

ExponentialMovingAverage::ExponentialMovingAverage(float decay, const torch::Device& device)
    : decay_(decay), device_(device)
{
    if (decay < 0.0f || decay > 1.0f) {
        throw std::invalid_argument("EMA decay must be between 0 and 1");
    }
}

ExponentialMovingAverage::~ExponentialMovingAverage() = default;

void ExponentialMovingAverage::RegisterModel(const torch::nn::Module& model, const std::string& prefix) {
    torch::NoGradGuard no_grad;
    
    std::unordered_map<std::string, torch::Tensor> model_params;
    CollectModelParameters(model, prefix, model_params);
    
    for (const auto& item : model_params) {
        RegisterParameter(item.first, item.second);
    }
}

void ExponentialMovingAverage::RegisterParameter(const std::string& name, const torch::Tensor& param) {
    torch::NoGradGuard no_grad;
    
    if (ema_params_.find(name) != ema_params_.end()) {
        // Parameter already registered, update shape if needed
        if (!ema_params_[name].sizes().equals(param.sizes())) {
            ema_params_[name] = param.detach().clone().to(device_).set_requires_grad(false);
        }
    } else {
        // New parameter - initialize with current value
        ema_params_[name] = param.detach().clone().to(device_).set_requires_grad(false);
    }
}

void ExponentialMovingAverage::Update(const torch::nn::Module& model, const std::string& prefix) {
    torch::NoGradGuard no_grad;
    
    std::unordered_map<std::string, torch::Tensor> model_params;
    CollectModelParameters(model, prefix, model_params);
    
    for (const auto& item : model_params) {
        UpdateParameter(item.first, item.second);
    }
}

void ExponentialMovingAverage::UpdateParameter(const std::string& name, const torch::Tensor& param) {
    torch::NoGradGuard no_grad;
    
    auto it = ema_params_.find(name);
    if (it == ema_params_.end()) {
        // Parameter not registered - register it first
        RegisterParameter(name, param);
        return;
    }
    
    // EMA update: ema = decay * ema + (1 - decay) * param
    // Use non-in-place operations to avoid gradient tracking issues
    torch::Tensor param_on_device = param.detach().to(device_).set_requires_grad(false);
    torch::Tensor new_ema = decay_ * it->second + (1.0f - decay_) * param_on_device;
    it->second = new_ema.set_requires_grad(false);
}

void ExponentialMovingAverage::ApplyToModel(torch::nn::Module& model, const std::string& prefix) const {
    torch::NoGradGuard no_grad;
    
    // Get current model parameters
    std::unordered_map<std::string, torch::Tensor> model_params;
    CollectModelParameters(model, prefix, model_params);
    
    // Apply EMA values to model parameters
    for (auto& item : model.named_parameters()) {
        const std::string& name = item.key();
        torch::Tensor& param = item.value();
        std::string full_name = GetParameterName(prefix, name);
        auto ema_it = ema_params_.find(full_name);
        
        if (ema_it != ema_params_.end()) {
            param.copy_(ema_it->second.to(param.device()));
        }
    }
}

torch::Tensor ExponentialMovingAverage::GetParameter(const std::string& name) const {
    auto it = ema_params_.find(name);
    if (it == ema_params_.end()) {
        throw std::runtime_error("EMA parameter '" + name + "' not found");
    }
    return it->second;
}

bool ExponentialMovingAverage::HasParameter(const std::string& name) const {
    return ema_params_.find(name) != ema_params_.end();
}

void ExponentialMovingAverage::Save(const std::string& path) const {
    std::vector<torch::Tensor> tensors;
    std::vector<std::string> names;
    
    // Save decay value
    tensors.push_back(torch::tensor(decay_));
    names.push_back("decay");
    
    // Save all EMA parameters
    for (const auto& item : ema_params_) {
        tensors.push_back(item.second.cpu());
        names.push_back(item.first);
    }
    
    torch::save(tensors, path);
    
    // Save names separately (torch::save doesn't handle strings well)
    std::ofstream names_file(path + ".names");
    for (const auto& name : names) {
        names_file << name << "\n";
    }
}

void ExponentialMovingAverage::Load(const std::string& path) {
    std::vector<torch::Tensor> tensors;
    torch::load(tensors, path);
    
    // Load names
    std::vector<std::string> names;
    std::ifstream names_file(path + ".names");
    std::string name;
    while (std::getline(names_file, name)) {
        names.push_back(name);
    }
    
    if (tensors.size() != names.size()) {
        throw std::runtime_error("EMA load failed: tensor count mismatch");
    }
    
    ema_params_.clear();
    
    // Load decay (first tensor)
    if (!names.empty() && names[0] == "decay") {
        decay_ = tensors[0].item<float>();
        
        // Load EMA parameters
        for (size_t i = 1; i < tensors.size(); ++i) {
            ema_params_[names[i]] = tensors[i].to(device_).set_requires_grad(false);
        }
    }
}

void ExponentialMovingAverage::Reset() {
    ema_params_.clear();
}

std::string ExponentialMovingAverage::GetParameterName(const std::string& prefix, const std::string& param_name) const {
    return prefix.empty() ? param_name : prefix + "." + param_name;
}

void ExponentialMovingAverage::CollectModelParameters(const torch::nn::Module& model, const std::string& prefix,
                                                     std::unordered_map<std::string, torch::Tensor>& params) const {
    for (const auto& item : model.named_parameters()) {
        const std::string& name = item.key();
        const torch::Tensor& param = item.value();
        std::string full_name = GetParameterName(prefix, name);
        params[full_name] = param;
    }
}

} // namespace utils
