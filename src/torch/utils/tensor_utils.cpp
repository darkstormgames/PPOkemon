#include "torch/utils/tensor_utils.h"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace utils {

float TensorUtils::GetGradientNorm(const std::vector<torch::Tensor>& parameters) {
    torch::Tensor total_norm = torch::zeros({});
    
    for (const auto& param : parameters) {
        if (param.grad().defined()) {
            torch::Tensor param_norm = param.grad().norm();
            total_norm += param_norm.pow(2);
        }
    }
    
    return std::sqrt(total_norm.item<float>());
}

void TensorUtils::ClipGradientNorm(const std::vector<torch::Tensor>& parameters, float max_norm) {
    torch::NoGradGuard no_grad;
    
    float total_norm = GetGradientNorm(parameters);
    float clip_coef = max_norm / (total_norm + 1e-6f);
    
    if (clip_coef < 1.0f) {
        for (const auto& param : parameters) {
            if (param.grad().defined()) {
                param.grad().mul_(clip_coef);
            }
        }
    }
}

void TensorUtils::ClipGradientValue(const std::vector<torch::Tensor>& parameters, float clip_value) {
    torch::NoGradGuard no_grad;
    
    for (const auto& param : parameters) {
        if (param.grad().defined()) {
            param.grad().clamp_(-clip_value, clip_value);
        }
    }
}

torch::Tensor TensorUtils::Standardize(const torch::Tensor& tensor, float epsilon) {
    torch::Tensor mean = tensor.mean();
    torch::Tensor std = tensor.std();
    return (tensor - mean) / (std + epsilon);
}

torch::Tensor TensorUtils::Normalize(const torch::Tensor& tensor, float min_val, float max_val) {
    torch::Tensor min_tensor = tensor.min();
    torch::Tensor max_tensor = tensor.max();
    torch::Tensor range = max_tensor - min_tensor;
    
    if (range.item<float>() < 1e-8f) {
        return torch::full_like(tensor, (min_val + max_val) / 2.0f);
    }
    
    return min_val + (max_val - min_val) * (tensor - min_tensor) / range;
}

torch::Tensor TensorUtils::DiscountedCumSum(const torch::Tensor& rewards, float gamma) {
    torch::Tensor discounted = torch::zeros_like(rewards);
    float running_sum = 0.0f;
    
    // Process rewards backwards
    auto rewards_accessor = rewards.accessor<float, 1>();
    auto discounted_accessor = discounted.accessor<float, 1>();
    
    for (int64_t i = rewards.size(0) - 1; i >= 0; --i) {
        running_sum = rewards_accessor[i] + gamma * running_sum;
        discounted_accessor[i] = running_sum;
    }
    
    return discounted;
}

torch::Tensor TensorUtils::ComputeGAE(const torch::Tensor& rewards, const torch::Tensor& values, 
                                     const torch::Tensor& dones, float gamma, float lambda) {
    int64_t n_steps = rewards.size(0);
    torch::Tensor advantages = torch::zeros_like(rewards);
    
    auto rewards_accessor = rewards.accessor<float, 1>();
    auto values_accessor = values.accessor<float, 1>();
    auto dones_accessor = dones.accessor<float, 1>();
    auto advantages_accessor = advantages.accessor<float, 1>();
    
    float gae = 0.0f;
    
    for (int64_t i = n_steps - 1; i >= 0; --i) {
        float next_value = (i == n_steps - 1) ? 0.0f : values_accessor[i + 1];
        float delta = rewards_accessor[i] + gamma * next_value * (1.0f - dones_accessor[i]) - values_accessor[i];
        gae = delta + gamma * lambda * (1.0f - dones_accessor[i]) * gae;
        advantages_accessor[i] = gae;
    }
    
    return advantages;
}

torch::Tensor TensorUtils::SampleFromLogits(const torch::Tensor& logits, bool deterministic) {
    if (deterministic) {
        return torch::argmax(logits, -1);
    } else {
        torch::Tensor probs = torch::softmax(logits, -1);
        return torch::multinomial(probs, 1).squeeze(-1);
    }
}

torch::Tensor TensorUtils::GetLogProbs(const torch::Tensor& logits, const torch::Tensor& actions) {
    torch::Tensor log_probs = torch::log_softmax(logits, -1);
    return log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1);
}

torch::Tensor TensorUtils::GetEntropy(const torch::Tensor& logits) {
    torch::Tensor probs = torch::softmax(logits, -1);
    torch::Tensor log_probs = torch::log_softmax(logits, -1);
    return -(probs * log_probs).sum(-1);
}

void TensorUtils::PrintTensorStats(const torch::Tensor& tensor, const std::string& name) {
    if (!tensor.defined()) {
        std::cout << name << ": undefined tensor" << std::endl;
        return;
    }
    
    std::cout << name << " - Shape: [";
    for (int64_t i = 0; i < tensor.dim(); ++i) {
        std::cout << tensor.size(i);
        if (i < tensor.dim() - 1) std::cout << ", ";
    }
    std::cout << "], ";
    
    if (tensor.numel() > 0) {
        std::cout << "Min: " << tensor.min().item<float>() 
                  << ", Max: " << tensor.max().item<float>()
                  << ", Mean: " << tensor.mean().item<float>()
                  << ", Std: " << tensor.std().item<float>();
    }
    
    std::cout << std::endl;
}

bool TensorUtils::CheckTensorValid(const torch::Tensor& tensor, const std::string& name) {
    if (!tensor.defined()) {
        std::cerr << "ERROR: " << name << " is undefined!" << std::endl;
        return false;
    }
    
    if (torch::any(torch::isnan(tensor)).item<bool>()) {
        std::cerr << "ERROR: " << name << " contains NaN values!" << std::endl;
        return false;
    }
    
    if (torch::any(torch::isinf(tensor)).item<bool>()) {
        std::cerr << "ERROR: " << name << " contains Inf values!" << std::endl;
        return false;
    }
    
    return true;
}

torch::Device TensorUtils::GetOptimalDevice() {
    if (torch::cuda::is_available()) {
        return torch::Device(torch::kCUDA, 0);
    }
    return torch::Device(torch::kCPU);
}

torch::Tensor TensorUtils::ToDevice(const torch::Tensor& tensor, const torch::Device& device) {
    if (tensor.device() == device) {
        return tensor;
    }
    return tensor.to(device);
}

std::vector<torch::Tensor> TensorUtils::SplitTensorBatch(const torch::Tensor& tensor, int64_t batch_size) {
    std::vector<torch::Tensor> batches;
    int64_t total_size = tensor.size(0);
    
    for (int64_t i = 0; i < total_size; i += batch_size) {
        int64_t end_idx = std::min(i + batch_size, total_size);
        batches.push_back(tensor.slice(0, i, end_idx));
    }
    
    return batches;
}

torch::Tensor TensorUtils::ConcatenateTensors(const std::vector<torch::Tensor>& tensors, int64_t dim) {
    if (tensors.empty()) {
        throw std::runtime_error("Cannot concatenate empty tensor list");
    }
    
    return torch::cat(tensors, dim);
}

} // namespace utils
