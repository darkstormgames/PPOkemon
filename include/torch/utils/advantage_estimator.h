#pragma once

#include <torch/torch.h>

namespace utils {

/**
 * Collection of advantage estimation methods for reinforcement learning
 */
class AdvantageEstimator {
public:
    // Generalized Advantage Estimation (GAE)
    static torch::Tensor ComputeGAE(const torch::Tensor& rewards, 
                                   const torch::Tensor& values, 
                                   const torch::Tensor& dones,
                                   float gamma = 0.99f, 
                                   float lambda = 0.95f);
    
    // TD(λ) advantage estimation
    static torch::Tensor ComputeTDLambda(const torch::Tensor& rewards,
                                        const torch::Tensor& values,
                                        const torch::Tensor& dones,
                                        float gamma = 0.99f,
                                        float lambda = 0.95f);
    
    // N-step advantage estimation
    static torch::Tensor ComputeNStepAdvantage(const torch::Tensor& rewards,
                                              const torch::Tensor& values,
                                              const torch::Tensor& dones,
                                              int n_steps = 5,
                                              float gamma = 0.99f);
    
    // Simple TD advantage (1-step)
    static torch::Tensor ComputeTDAdvantage(const torch::Tensor& rewards,
                                           const torch::Tensor& values,
                                           const torch::Tensor& dones,
                                           float gamma = 0.99f);
    
    // Monte Carlo returns (for comparison)
    static torch::Tensor ComputeMonteCarloReturns(const torch::Tensor& rewards,
                                                  const torch::Tensor& dones,
                                                  float gamma = 0.99f);
    
    // Discounted cumulative sum utility
    static torch::Tensor DiscountedCumSum(const torch::Tensor& x, 
                                         const torch::Tensor& dones,
                                         float gamma = 0.99f);
    
    // Value targets for training value function
    static torch::Tensor ComputeValueTargets(const torch::Tensor& rewards,
                                            const torch::Tensor& values,
                                            const torch::Tensor& dones,
                                            float gamma = 0.99f,
                                            float lambda = 0.95f);
    
    // Advantage normalization utilities
    static torch::Tensor NormalizeAdvantages(const torch::Tensor& advantages, 
                                            float epsilon = 1e-8f);
    
    static torch::Tensor ClipAdvantages(const torch::Tensor& advantages,
                                       float clip_value = 10.0f);

private:
    // Helper function for recursive advantage computation
    static void ComputeGAERecursive(torch::Tensor& advantages,
                                   const torch::Tensor& rewards,
                                   const torch::Tensor& values,
                                   const torch::Tensor& dones,
                                   float gamma,
                                   float lambda);
};

} // namespace utils
