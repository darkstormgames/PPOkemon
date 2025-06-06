#include <torch/networks/a2c.h>

namespace networks {

// A2C Implementation

// Constructor with shared backbone
A2CImpl::A2CImpl(SharedBackboneTag, std::shared_ptr<NetworkBase> backbone, int64_t action_dim, int64_t value_dim)
    : backbone_(backbone), action_dim_(action_dim), value_dim_(value_dim)
{
    InitializeHeads();
}

// Constructor for CNN-based A2C
A2CImpl::A2CImpl(CNNTag, int64_t num_input_channels, int64_t input_height, int64_t input_width, 
                 int64_t action_dim, int64_t value_dim, int64_t conv_out_size)
    : action_dim_(action_dim), value_dim_(value_dim)
{
    // Create CNN backbone
    auto cnn_backbone = std::make_shared<CNNBodyImpl>(num_input_channels, input_height, input_width, conv_out_size);
    backbone_ = cnn_backbone;
    InitializeHeads();
}

// Constructor for MLP-based A2C
A2CImpl::A2CImpl(MLPTag, int64_t input_dim, int64_t hidden_dim, int64_t action_dim, int64_t value_dim)
    : action_dim_(action_dim), value_dim_(value_dim)
{
    // Create MLP backbone
    auto mlp_backbone = std::make_shared<MLPImpl>(input_dim, hidden_dim, hidden_dim);
    backbone_ = mlp_backbone;
    InitializeHeads();
}

A2CImpl::~A2CImpl()
{
}

torch::Tensor A2CImpl::forward(const torch::Tensor &input)
{
    // Default forward returns actor output
    return GetAction(input);
}

std::tuple<torch::Tensor, torch::Tensor> A2CImpl::ForwardActorCritic(const torch::Tensor& input)
{
    torch::Tensor features = backbone_->forward(input);
    torch::Tensor action_logits = actor_head_(features);
    torch::Tensor value = critic_head_(features);
    return std::make_tuple(action_logits, value);
}

torch::Tensor A2CImpl::GetAction(const torch::Tensor& input)
{
    torch::Tensor features = backbone_->forward(input);
    return actor_head_(features);
}

torch::Tensor A2CImpl::GetValue(const torch::Tensor& input)
{
    torch::Tensor features = backbone_->forward(input);
    return critic_head_(features);
}

void A2CImpl::InitOrtho(const float gain)
{
    // Initialize backbone
    backbone_->InitOrtho(gain);
    
    // Initialize heads
    torch::NoGradGuard no_grad;
    torch::nn::init::orthogonal_(actor_head_->weight, gain);
    torch::nn::init::constant_(actor_head_->bias, 0.0);
    torch::nn::init::orthogonal_(critic_head_->weight, gain);
    torch::nn::init::constant_(critic_head_->bias, 0.0);
}

void A2CImpl::InitializeHeads()
{
    int64_t backbone_output_size = backbone_->GetOutputSize();
    if (backbone_output_size <= 0) {
        throw std::runtime_error("Invalid backbone output size for A2C network");
    }
    
    actor_head_ = register_module("actor_head", torch::nn::Linear(backbone_output_size, action_dim_));
    critic_head_ = register_module("critic_head", torch::nn::Linear(backbone_output_size, value_dim_));
}

} // namespace networks