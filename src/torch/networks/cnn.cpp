#include <torch/networks/cnn.h>
#include <cmath>

namespace networks {

// CNNBody Implementation
CNNBodyImpl::CNNBodyImpl(int64_t num_input_channels, int64_t input_height, int64_t input_width, int64_t conv_out_size)
    : input_height_(input_height), input_width_(input_width), output_size_(conv_out_size)
{
    // Adjusted CNN architecture for Gameboy screen resolution (160x144)
    // Input: (Batch, Channels, Height, Width)

    // Layer 1
    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(num_input_channels, 32, /*kernel_size=*/8).stride(4).padding(2))); // Adjusted padding
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(32));

    // Layer 2
    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, /*kernel_size=*/4).stride(2).padding(1))); // Adjusted padding
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(64));

    // Layer 3
    conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, /*kernel_size=*/3).stride(1).padding(1))); // Adjusted padding
    bn3 = register_module("bn3", torch::nn::BatchNorm2d(64));

    // Calculate the flattened size after convolutions
    // Helper function to calculate output size of a convolutional layer
    auto conv_output_size = [](int64_t size, int64_t kernel_size, int64_t stride, int64_t padding) {
        return (int64_t)std::floor((size - kernel_size + 2 * padding) / stride + 1);
    };

    int64_t h = input_height_;
    int64_t w = input_width_;

    h = conv_output_size(h, 8, 4, 2); // conv1
    w = conv_output_size(w, 8, 4, 2);
    h = conv_output_size(h, 4, 2, 1); // conv2
    w = conv_output_size(w, 4, 2, 1);
    h = conv_output_size(h, 3, 1, 1); // conv3
    w = conv_output_size(w, 3, 1, 1);

    const int64_t flattened_size = 64 * h * w;
    linear = register_module("linear", torch::nn::Linear(flattened_size, conv_out_size));

    InitOrtho(std::sqrt(2.0f)); // Initialize weights
}

CNNBodyImpl::~CNNBodyImpl()
{
}

torch::Tensor CNNBodyImpl::forward(const torch::Tensor &in)
{
    // Ensure input is in NCHW format
    // Original Gameboy screen is 160x144. If input is (N, H, W), permute to (N, 1, H, W) if single channel
    // Or if it's already (N, C, H, W), it's fine.
    // For this example, let's assume `in` is (N, C, H, W)
    // E.g. if obs is (Batch, 160, 144) -> .unsqueeze(1) to make it (Batch, 1, 160, 144)
    // If obs is (Batch, 3, 160, 144) for stacked frames, it's fine.

    torch::Tensor x = torch::relu(bn1(conv1(in)));
    x = torch::relu(bn2(conv2(x)));
    x = torch::relu(bn3(conv3(x)));
    x = x.view({x.size(0), -1}); // Flatten
    x = torch::relu(linear(x));
    return x;
}

void CNNBodyImpl::InitOrtho(const float gain)
{
    torch::NoGradGuard no_grad;
    auto init_layer = [&](torch::nn::Module &layer)
    {
        if (auto *conv = layer.as<torch::nn::Conv2d>())
        {
            torch::nn::init::orthogonal_(conv->weight, gain);
            if (conv->bias.defined())
            {
                torch::nn::init::constant_(conv->bias, 0.0);
            }
        }
        else if (auto *lin = layer.as<torch::nn::Linear>())
        {
            torch::nn::init::orthogonal_(lin->weight, gain);
            if (lin->bias.defined())
            {
                torch::nn::init::constant_(lin->bias, 0.0);
            }
        }
    };
    init_layer(*conv1);
    init_layer(*conv2);
    init_layer(*conv3);
    init_layer(*linear);
    // Initialize BatchNorm layers
    if (auto *bn = bn1.get())
    {
        torch::nn::init::constant_(bn->weight, 1.0);
        torch::nn::init::constant_(bn->bias, 0.0);
    }
    if (auto *bn = bn2.get())
    {
        torch::nn::init::constant_(bn->weight, 1.0);
        torch::nn::init::constant_(bn->bias, 0.0);
    }
    if (auto *bn = bn3.get())
    {
        torch::nn::init::constant_(bn->weight, 1.0);
        torch::nn::init::constant_(bn->bias, 0.0);
    }
}

} // namespace networks