#pragma once

#include <torch/torch.h>
#include <string>

namespace utils {

/**
 * Weight initialization utilities for neural networks
 * Implements various initialization schemes commonly used in deep RL
 */
class WeightInitializer {
public:
    enum class InitType {
        Xavier,          // Xavier/Glorot uniform/normal
        Kaiming,         // He initialization
        Orthogonal,      // Orthogonal initialization (common in RL)
        Uniform,         // Uniform initialization
        Normal,          // Normal initialization
        Zero,            // Zero initialization
        One,             // One initialization
        Constant         // Constant value initialization
    };
    
    enum class Activation {
        Linear,
        ReLU,
        LeakyReLU,
        Tanh,
        Sigmoid,
        SELU
    };
    
    // Initialize a single tensor
    static void InitializeTensor(torch::Tensor& tensor, 
                                InitType type = InitType::Xavier,
                                Activation activation = Activation::ReLU,
                                float gain = 1.0f,
                                float constant_value = 0.0f);
    
    // Initialize all parameters in a module
    static void InitializeModule(torch::nn::Module& module,
                                InitType type = InitType::Orthogonal,
                                Activation activation = Activation::ReLU,
                                float gain = 1.0f);
    
    // Specialized initializations for common layer types
    static void InitializeLinear(torch::nn::Linear& linear,
                                InitType type = InitType::Orthogonal,
                                float gain = 1.0f,
                                bool init_bias = true,
                                float bias_value = 0.0f);
    
    static void InitializeConv2d(torch::nn::Conv2d& conv,
                                InitType type = InitType::Kaiming,
                                Activation activation = Activation::ReLU,
                                bool init_bias = true,
                                float bias_value = 0.0f);
    
    // Policy-specific initializations (common in RL)
    static void InitializePolicyHead(torch::nn::Linear& policy_head,
                                    float std = 0.01f);
    
    static void InitializeValueHead(torch::nn::Linear& value_head,
                                   float std = 1.0f);
    
    // LSTM/GRU initialization
    static void InitializeLSTM(torch::nn::LSTM& lstm,
                              InitType type = InitType::Orthogonal,
                              float gain = 1.0f);
    
    static void InitializeGRU(torch::nn::GRU& gru,
                             InitType type = InitType::Orthogonal,
                             float gain = 1.0f);
    
    // Utility functions
    static float CalculateGain(Activation activation, float param = 0.0f);
    static float CalculateFanIn(const torch::Tensor& tensor);
    static float CalculateFanOut(const torch::Tensor& tensor);
    static float CalculateFanAvg(const torch::Tensor& tensor);
    
    // Custom initialization for attention mechanisms
    static void InitializeAttention(torch::nn::Module& attention_module,
                                   float scale = 1.0f);
    
    // Batch norm initialization
    static void InitializeBatchNorm(torch::nn::BatchNorm2d& bn,
                                   float weight_value = 1.0f,
                                   float bias_value = 0.0f);
    
    // Layer norm initialization  
    static void InitializeLayerNorm(torch::nn::LayerNorm& ln,
                                   float weight_value = 1.0f,
                                   float bias_value = 0.0f);

private:
    // Helper functions for different initialization types
    static void XavierUniform(torch::Tensor& tensor, float gain = 1.0f);
    static void XavierNormal(torch::Tensor& tensor, float gain = 1.0f);
    static void KaimingUniform(torch::Tensor& tensor, Activation activation = Activation::ReLU);
    static void KaimingNormal(torch::Tensor& tensor, Activation activation = Activation::ReLU);
    static void OrthogonalInit(torch::Tensor& tensor, float gain = 1.0f);
};

} // namespace utils
