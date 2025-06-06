#pragma once

#include "network_base.h"
#include <torch/torch.h>

namespace networks {

struct MLPImpl : public NetworkBase {
    // Constructor for a generic MLP
    MLPImpl(const int64_t num_in, const int64_t num_hidden, const int64_t out_num);
    ~MLPImpl();

    // Override NetworkBase methods
    torch::Tensor forward(const torch::Tensor& input) override;
    void InitOrtho(const float gain_backbone = 1.0f, const float gain_out = 1.0f);
    int64_t GetOutputSize() const override { return output_size_; }

    // Network layers
    torch::nn::Linear l1{nullptr};
    torch::nn::Linear l2{nullptr};
    torch::nn::Linear out_layer{nullptr};

private:
    int64_t output_size_;
};
TORCH_MODULE(MLP);

} // namespace networks
