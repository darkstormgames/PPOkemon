#include "torch/utils/advantage_estimator.h"
#include <iostream>

namespace utils {

torch::Tensor AdvantageEstimator::ComputeGAE(const torch::Tensor& rewards, 
                                            const torch::Tensor& values, 
                                            const torch::Tensor& dones,
                                            float gamma, 
                                            float lambda) {
    torch::NoGradGuard no_grad;
    
    const int64_t seq_len = rewards.size(0);
    torch::Tensor advantages = torch::zeros_like(rewards);
    
    // Convert to CPU for easier access (can optimize later)
    auto rewards_cpu = rewards.cpu();
    auto values_cpu = values.cpu();
    auto dones_cpu = dones.cpu();
    auto rewards_accessor = rewards_cpu.accessor<float, 1>();
    auto values_accessor = values_cpu.accessor<float, 1>();
    auto dones_accessor = dones_cpu.accessor<float, 1>();
    auto advantages_accessor = advantages.accessor<float, 1>();
    
    float gae = 0.0f;
    
    // Compute GAE backwards through time
    for (int64_t t = seq_len - 1; t >= 0; --t) {
        float next_value = (t == seq_len - 1) ? 0.0f : values_accessor[t + 1];
        float delta = rewards_accessor[t] + gamma * next_value * (1.0f - dones_accessor[t]) - values_accessor[t];
        gae = delta + gamma * lambda * (1.0f - dones_accessor[t]) * gae;
        advantages_accessor[t] = gae;
    }
    
    return advantages.to(rewards.device());
}

torch::Tensor AdvantageEstimator::ComputeTDLambda(const torch::Tensor& rewards,
                                                 const torch::Tensor& values,
                                                 const torch::Tensor& dones,
                                                 float gamma,
                                                 float lambda) {
    torch::NoGradGuard no_grad;
    
    const int64_t seq_len = rewards.size(0);
    torch::Tensor returns = torch::zeros_like(rewards);
    
    auto rewards_cpu = rewards.cpu();
    auto values_cpu = values.cpu();
    auto dones_cpu = dones.cpu();
    auto rewards_accessor = rewards_cpu.accessor<float, 1>();
    auto values_accessor = values_cpu.accessor<float, 1>();
    auto dones_accessor = dones_cpu.accessor<float, 1>();
    auto returns_accessor = returns.accessor<float, 1>();
    
    float td_return = 0.0f;
    
    // Compute TD(λ) returns backwards
    for (int64_t t = seq_len - 1; t >= 0; --t) {
        float next_value = (t == seq_len - 1) ? 0.0f : values_accessor[t + 1];
        float td_target = rewards_accessor[t] + gamma * next_value * (1.0f - dones_accessor[t]);
        
        if (t == seq_len - 1) {
            td_return = td_target;
        } else {
            td_return = td_target + gamma * lambda * (1.0f - dones_accessor[t]) * 
                       (td_return - next_value);
        }
        
        returns_accessor[t] = td_return;
    }
    
    // Advantages = returns - values
    return (returns - values).to(rewards.device());
}

torch::Tensor AdvantageEstimator::ComputeNStepAdvantage(const torch::Tensor& rewards,
                                                       const torch::Tensor& values,
                                                       const torch::Tensor& dones,
                                                       int n_steps,
                                                       float gamma) {
    torch::NoGradGuard no_grad;
    
    const int64_t seq_len = rewards.size(0);
    torch::Tensor advantages = torch::zeros_like(rewards);
    
    auto rewards_cpu = rewards.cpu();
    auto values_cpu = values.cpu();
    auto dones_cpu = dones.cpu();
    auto rewards_accessor = rewards_cpu.accessor<float, 1>();
    auto values_accessor = values_cpu.accessor<float, 1>();
    auto dones_accessor = dones_cpu.accessor<float, 1>();
    auto advantages_accessor = advantages.accessor<float, 1>();
    
    for (int64_t t = 0; t < seq_len; ++t) {
        float n_step_return = 0.0f;
        float discount = 1.0f;
        bool terminated = false;
        
        // Accumulate n-step return
        for (int step = 0; step < n_steps && (t + step) < seq_len; ++step) {
            if (terminated) break;
            
            n_step_return += discount * rewards_accessor[t + step];
            discount *= gamma;
            
            if (dones_accessor[t + step] > 0.5f) {
                terminated = true;
            }
        }
        
        // Add bootstrap value if not terminated
        if (!terminated && (t + n_steps) < seq_len) {
            n_step_return += discount * values_accessor[t + n_steps];
        }
        
        advantages_accessor[t] = n_step_return - values_accessor[t];
    }
    
    return advantages.to(rewards.device());
}

torch::Tensor AdvantageEstimator::ComputeTDAdvantage(const torch::Tensor& rewards,
                                                    const torch::Tensor& values,
                                                    const torch::Tensor& dones,
                                                    float gamma) {
    torch::NoGradGuard no_grad;
    
    const int64_t seq_len = rewards.size(0);
    
    // Compute next values (shifted by 1, with 0 at the end)
    torch::Tensor next_values = torch::zeros_like(values);
    if (seq_len > 1) {
        next_values.slice(0, 0, seq_len - 1) = values.slice(0, 1, seq_len);
    }
    
    // TD target = reward + gamma * next_value * (1 - done)
    torch::Tensor td_targets = rewards + gamma * next_values * (1.0f - dones);
    
    // Advantage = TD target - current value
    return td_targets - values;
}

torch::Tensor AdvantageEstimator::ComputeMonteCarloReturns(const torch::Tensor& rewards,
                                                          const torch::Tensor& dones,
                                                          float gamma) {
    return DiscountedCumSum(rewards, dones, gamma);
}

torch::Tensor AdvantageEstimator::DiscountedCumSum(const torch::Tensor& x, 
                                                  const torch::Tensor& dones,
                                                  float gamma) {
    torch::NoGradGuard no_grad;
    
    const int64_t seq_len = x.size(0);
    torch::Tensor result = torch::zeros_like(x);
    
    auto x_cpu = x.cpu();
    auto dones_cpu = dones.cpu();
    auto x_accessor = x_cpu.accessor<float, 1>();
    auto dones_accessor = dones_cpu.accessor<float, 1>();
    auto result_accessor = result.accessor<float, 1>();
    
    float running_sum = 0.0f;
    
    // Process backwards
    for (int64_t t = seq_len - 1; t >= 0; --t) {
        running_sum = x_accessor[t] + gamma * running_sum * (1.0f - dones_accessor[t]);
        result_accessor[t] = running_sum;
    }
    
    return result.to(x.device());
}

torch::Tensor AdvantageEstimator::ComputeValueTargets(const torch::Tensor& rewards,
                                                     const torch::Tensor& values,
                                                     const torch::Tensor& dones,
                                                     float gamma,
                                                     float lambda) {
    torch::NoGradGuard no_grad;
    
    // Compute GAE advantages
    torch::Tensor advantages = ComputeGAE(rewards, values, dones, gamma, lambda);
    
    // Value targets = advantages + values
    return advantages + values;
}

torch::Tensor AdvantageEstimator::NormalizeAdvantages(const torch::Tensor& advantages, 
                                                     float epsilon) {
    torch::NoGradGuard no_grad;
    
    if (advantages.numel() == 0) {
        return advantages;
    }
    
    torch::Tensor mean = advantages.mean();
    torch::Tensor std = advantages.std();
    
    return (advantages - mean) / (std + epsilon);
}

torch::Tensor AdvantageEstimator::ClipAdvantages(const torch::Tensor& advantages,
                                                float clip_value) {
    return torch::clamp(advantages, -clip_value, clip_value);
}

void AdvantageEstimator::ComputeGAERecursive(torch::Tensor& advantages,
                                            const torch::Tensor& rewards,
                                            const torch::Tensor& values,
                                            const torch::Tensor& dones,
                                            float gamma,
                                            float lambda) {
    // This is an alternative implementation that could be optimized further
    // Current implementation uses the direct approach above
    // Suppress unused parameter warnings for now
    (void)advantages; (void)rewards; (void)values; (void)dones; (void)gamma; (void)lambda;
}

} // namespace utils
