#pragma once

#include <torch/torch.h>

namespace networks {

// Abstract base class for all DRL networks
class NetworkBase : public torch::nn::Module {
public:
    virtual ~NetworkBase() = default;
    
    // Pure virtual function for forward pass
    virtual torch::Tensor forward(const torch::Tensor& input) = 0;
    
    // Virtual function for orthogonal initialization (default implementation)
    virtual void InitOrtho(const float gain = 1.0f) { (void)gain; }
    
    // Optional: Get output size/features
    virtual int64_t GetOutputSize() const { return -1; }
    
    // Optional: Reset/reinitialize the network
    virtual void Reset() {}
};

} // namespace networks
