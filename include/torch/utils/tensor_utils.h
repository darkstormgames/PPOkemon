#pragma once

#include <torch/torch.h>
#include <vector>

namespace utils {

class TensorUtils {
public:
    // Gradient utilities
    static float GetGradientNorm(const std::vector<torch::Tensor>& parameters);
    static void ClipGradientNorm(const std::vector<torch::Tensor>& parameters, float max_norm);
    static void ClipGradientValue(const std::vector<torch::Tensor>& parameters, float clip_value);
    
    // Tensor manipulation
    static torch::Tensor Standardize(const torch::Tensor& tensor, float epsilon = 1e-8f);
    static torch::Tensor Normalize(const torch::Tensor& tensor, float min_val = 0.0f, float max_val = 1.0f);
    static torch::Tensor DiscountedCumSum(const torch::Tensor& rewards, float gamma);
    
    // GAE (Generalized Advantage Estimation) computation
    static torch::Tensor ComputeGAE(const torch::Tensor& rewards, const torch::Tensor& values, 
                                   const torch::Tensor& dones, float gamma = 0.99f, float lambda = 0.95f);
    
    // Action utilities
    static torch::Tensor SampleFromLogits(const torch::Tensor& logits, bool deterministic = false);
    static torch::Tensor GetLogProbs(const torch::Tensor& logits, const torch::Tensor& actions);
    static torch::Tensor GetEntropy(const torch::Tensor& logits);
    
    // Debugging utilities
    static void PrintTensorStats(const torch::Tensor& tensor, const std::string& name);
    static bool CheckTensorValid(const torch::Tensor& tensor, const std::string& name = "");
    
    // Device utilities
    static torch::Device GetOptimalDevice();
    static torch::Tensor ToDevice(const torch::Tensor& tensor, const torch::Device& device);
    
    // Batch processing
    static std::vector<torch::Tensor> SplitTensorBatch(const torch::Tensor& tensor, int64_t batch_size);
    static torch::Tensor ConcatenateTensors(const std::vector<torch::Tensor>& tensors, int64_t dim = 0);
};

} // namespace utils
