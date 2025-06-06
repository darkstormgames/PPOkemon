#pragma once

#include "network_base.h"
#include <torch/torch.h>

namespace networks {

struct CNNBodyImpl : torch::nn::Module, public NetworkBase {
    // Constructor for CNN feature extractor optimized for Gameboy screen (160x144)
    CNNBodyImpl(int64_t num_input_channels, int64_t input_height, int64_t input_width, int64_t conv_out_size = 512);
    ~CNNBodyImpl();

    // Override NetworkBase methods
    torch::Tensor forward(const torch::Tensor& input) override;
    void InitOrtho(const float gain = 1.0f) override;
    int64_t GetOutputSize() const override { return output_size_; }

    // CNN layers
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::BatchNorm2d bn2{nullptr};
    torch::nn::Conv2d conv3{nullptr};
    torch::nn::BatchNorm2d bn3{nullptr};
    torch::nn::Linear linear{nullptr};

private:
    int64_t input_height_;
    int64_t input_width_;
    int64_t output_size_;
};
TORCH_MODULE(CNNBody);

} // namespace networks
