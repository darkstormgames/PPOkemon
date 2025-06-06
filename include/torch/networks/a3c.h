#pragma once

#include "a2c.h"
#include <torch/torch.h>

namespace networks {

// A3C (Asynchronous Advantage Actor-Critic) Network
// Uses composition instead of inheritance to avoid constructor ambiguity
struct A3CImpl : torch::nn::Module, public NetworkBase {
    // Constructor with shared backbone
    A3CImpl(SharedBackboneTag, std::shared_ptr<NetworkBase> backbone, int64_t action_dim, int64_t value_dim = 1);
    
    // Constructor for CNN-based A3C (for visual inputs)
    A3CImpl(CNNTag, int64_t num_input_channels, int64_t input_height, int64_t input_width, 
            int64_t action_dim, int64_t value_dim = 1, int64_t conv_out_size = 512);
    
    // Constructor for MLP-based A3C (for feature inputs)
    A3CImpl(MLPTag, int64_t input_dim, int64_t hidden_dim, int64_t action_dim, int64_t value_dim = 1);
    
    ~A3CImpl();

    // Override NetworkBase methods
    torch::Tensor forward(const torch::Tensor& input) override;
    void InitOrtho(const float gain = 1.0f) override;
    
    // A3C specific methods (delegate to internal A2C)
    std::tuple<torch::Tensor, torch::Tensor> ForwardActorCritic(const torch::Tensor& input);
    torch::Tensor GetAction(const torch::Tensor& input);
    torch::Tensor GetValue(const torch::Tensor& input);

    // Static factory methods for easier construction
    static std::shared_ptr<A3CImpl> WithSharedBackbone(std::shared_ptr<NetworkBase> backbone, int64_t action_dim, int64_t value_dim = 1) {
        return std::make_shared<A3CImpl>(SharedBackboneTag{}, backbone, action_dim, value_dim);
    }
    
    static std::shared_ptr<A3CImpl> WithCNN(int64_t num_input_channels, int64_t input_height, int64_t input_width, 
                       int64_t action_dim, int64_t value_dim = 1, int64_t conv_out_size = 512) {
        return std::make_shared<A3CImpl>(CNNTag{}, num_input_channels, input_height, input_width, action_dim, value_dim, conv_out_size);
    }
    
    static std::shared_ptr<A3CImpl> WithMLP(int64_t input_dim, int64_t hidden_dim, int64_t action_dim, int64_t value_dim = 1) {
        return std::make_shared<A3CImpl>(MLPTag{}, input_dim, hidden_dim, action_dim, value_dim);
    }

private:
    // Internal A2C network - using composition instead of inheritance
    A2C a2c_network_;
};
TORCH_MODULE(A3C);

} // namespace networks
