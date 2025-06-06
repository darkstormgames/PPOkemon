#pragma once

#include "network_base.h"
#include "mlp.h"
#include "cnn.h"
#include <torch/torch.h>

namespace networks {

// Tag types for constructor disambiguation
struct SharedBackboneTag {};
struct CNNTag {};
struct MLPTag {};

// A2C (Advantage Actor-Critic) Network
struct A2CImpl : torch::nn::Module, public NetworkBase {
    // Constructor with shared backbone
    A2CImpl(SharedBackboneTag, std::shared_ptr<NetworkBase> backbone, int64_t action_dim, int64_t value_dim = 1);
    
    // Constructor for CNN-based A2C (for visual inputs)
    A2CImpl(CNNTag, int64_t num_input_channels, int64_t input_height, int64_t input_width, 
            int64_t action_dim, int64_t value_dim = 1, int64_t conv_out_size = 512);
    
    // Constructor for MLP-based A2C (for feature inputs)
    A2CImpl(MLPTag, int64_t input_dim, int64_t hidden_dim, int64_t action_dim, int64_t value_dim = 1);
    
    ~A2CImpl();

    // Override NetworkBase methods
    torch::Tensor forward(const torch::Tensor& input) override;
    void InitOrtho(const float gain = 1.0f) override;
    
    // A2C specific methods
    std::tuple<torch::Tensor, torch::Tensor> ForwardActorCritic(const torch::Tensor& input);
    torch::Tensor GetAction(const torch::Tensor& input);
    torch::Tensor GetValue(const torch::Tensor& input);

    // Static factory methods for easier construction
    static std::shared_ptr<A2CImpl> WithSharedBackbone(std::shared_ptr<NetworkBase> backbone, int64_t action_dim, int64_t value_dim = 1) {
        return std::make_shared<A2CImpl>(SharedBackboneTag{}, backbone, action_dim, value_dim);
    }
    
    static std::shared_ptr<A2CImpl> WithCNN(int64_t num_input_channels, int64_t input_height, int64_t input_width, 
                       int64_t action_dim, int64_t value_dim = 1, int64_t conv_out_size = 512) {
        return std::make_shared<A2CImpl>(CNNTag{}, num_input_channels, input_height, input_width, action_dim, value_dim, conv_out_size);
    }
    
    static std::shared_ptr<A2CImpl> WithMLP(int64_t input_dim, int64_t hidden_dim, int64_t action_dim, int64_t value_dim = 1) {
        return std::make_shared<A2CImpl>(MLPTag{}, input_dim, hidden_dim, action_dim, value_dim);
    }

    // Network components
    std::shared_ptr<NetworkBase> backbone_;
    torch::nn::Linear actor_head_{nullptr};
    torch::nn::Linear critic_head_{nullptr};

private:
    int64_t action_dim_;
    int64_t value_dim_;
    void InitializeHeads();
};
TORCH_MODULE(A2C);

} // namespace networks
