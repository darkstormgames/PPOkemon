#include <torch/networks/a3c.h>

namespace networks {

// A3C Implementation using composition

// Constructor with shared backbone
A3CImpl::A3CImpl(SharedBackboneTag, std::shared_ptr<NetworkBase> backbone, int64_t action_dim, int64_t value_dim)
    : a2c_network_(A2C(SharedBackboneTag{}, backbone, action_dim, value_dim))
{
    // Register the A2C network as a submodule
    register_module("a2c", a2c_network_);
}

// Constructor for CNN-based A3C
A3CImpl::A3CImpl(CNNTag, int64_t num_input_channels, int64_t input_height, int64_t input_width, 
                 int64_t action_dim, int64_t value_dim, int64_t conv_out_size)
    : a2c_network_(A2C(CNNTag{}, num_input_channels, input_height, input_width, action_dim, value_dim, conv_out_size))
{
    // Register the A2C network as a submodule
    register_module("a2c", a2c_network_);
}

// Constructor for MLP-based A3C
A3CImpl::A3CImpl(MLPTag, int64_t input_dim, int64_t hidden_dim, int64_t action_dim, int64_t value_dim)
    : a2c_network_(A2C(MLPTag{}, input_dim, hidden_dim, action_dim, value_dim))
{
    // Register the A2C network as a submodule
    register_module("a2c", a2c_network_);
}

A3CImpl::~A3CImpl()
{
}

// Override NetworkBase methods - delegate to A2C
torch::Tensor A3CImpl::forward(const torch::Tensor& input)
{
    return a2c_network_->forward(input);
}

void A3CImpl::InitOrtho(const float gain)
{
    a2c_network_->InitOrtho(gain);
}

// A3C specific methods - delegate to A2C
std::tuple<torch::Tensor, torch::Tensor> A3CImpl::ForwardActorCritic(const torch::Tensor& input)
{
    return a2c_network_->ForwardActorCritic(input);
}

torch::Tensor A3CImpl::GetAction(const torch::Tensor& input)
{
    return a2c_network_->GetAction(input);
}

torch::Tensor A3CImpl::GetValue(const torch::Tensor& input)
{
    return a2c_network_->GetValue(input);
}

} // namespace networks