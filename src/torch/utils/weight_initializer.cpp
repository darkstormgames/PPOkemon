#include "torch/utils/weight_initializer.h"
#include <cmath>
#include <stdexcept>

namespace utils {

void WeightInitializer::InitializeTensor(torch::Tensor& tensor, 
                                        InitType type,
                                        Activation activation,
                                        float gain,
                                        float constant_value) {
    torch::NoGradGuard no_grad;
    
    if (gain <= 0.0f && type != InitType::Constant && type != InitType::Zero) {
        gain = CalculateGain(activation);
    }
    
    switch (type) {
        case InitType::Xavier:
            XavierNormal(tensor, gain);
            break;
        case InitType::Kaiming:
            KaimingNormal(tensor, activation);
            break;
        case InitType::Orthogonal:
            OrthogonalInit(tensor, gain);
            break;
        case InitType::Uniform:
            {
                float bound = std::sqrt(3.0f / CalculateFanIn(tensor)) * gain;
                torch::nn::init::uniform_(tensor, -bound, bound);
            }
            break;
        case InitType::Normal:
            {
                float std = gain / std::sqrt(CalculateFanIn(tensor));
                torch::nn::init::normal_(tensor, 0.0f, std);
            }
            break;
        case InitType::Zero:
            torch::nn::init::zeros_(tensor);
            break;
        case InitType::One:
            torch::nn::init::ones_(tensor);
            break;
        case InitType::Constant:
            torch::nn::init::constant_(tensor, constant_value);
            break;
    }
}

void WeightInitializer::InitializeModule(torch::nn::Module& module,
                                        InitType type,
                                        Activation activation,
                                        float gain) {
    torch::NoGradGuard no_grad;
    
    for (auto& param : module.named_parameters()) {
        // const std::string& name = param.key();  // Currently unused but may be needed for selective initialization
        torch::Tensor& tensor = param.value();
        
        // Skip 1D tensors (typically biases) for most initializations
        if (tensor.dim() == 1 && type != InitType::Constant && type != InitType::Zero) {
            torch::nn::init::zeros_(tensor);
        } else if (tensor.dim() >= 2) {
            InitializeTensor(tensor, type, activation, gain);
        }
    }
}

void WeightInitializer::InitializeLinear(torch::nn::Linear& linear,
                                        InitType type,
                                        float gain,
                                        bool init_bias,
                                        float bias_value) {
    torch::NoGradGuard no_grad;
    
    // Initialize weight
    InitializeTensor(linear->weight, type, Activation::Linear, gain);
    
    // Initialize bias
    if (init_bias && linear->bias.defined()) {
        torch::nn::init::constant_(linear->bias, bias_value);
    }
}

void WeightInitializer::InitializeConv2d(torch::nn::Conv2d& conv,
                                         InitType type,
                                         Activation activation,
                                         bool init_bias,
                                         float bias_value) {
    torch::NoGradGuard no_grad;
    
    // Initialize weight
    InitializeTensor(conv->weight, type, activation);
    
    // Initialize bias
    if (init_bias && conv->bias.defined()) {
        torch::nn::init::constant_(conv->bias, bias_value);
    }
}

void WeightInitializer::InitializePolicyHead(torch::nn::Linear& policy_head,
                                            float std) {
    torch::NoGradGuard no_grad;
    
    // Small initialization for policy head to ensure initial actions are near uniform
    torch::nn::init::normal_(policy_head->weight, 0.0f, std);
    if (policy_head->bias.defined()) {
        torch::nn::init::zeros_(policy_head->bias);
    }
}

void WeightInitializer::InitializeValueHead(torch::nn::Linear& value_head,
                                           float std) {
    torch::NoGradGuard no_grad;
    
    // Standard initialization for value head
    torch::nn::init::normal_(value_head->weight, 0.0f, std);
    if (value_head->bias.defined()) {
        torch::nn::init::zeros_(value_head->bias);
    }
}

void WeightInitializer::InitializeLSTM(torch::nn::LSTM& lstm,
                                      InitType type,
                                      float gain) {
    torch::NoGradGuard no_grad;
    
    for (auto& param : lstm->named_parameters()) {
        const std::string& name = param.key();
        torch::Tensor& tensor = param.value();
        
        if (name.find("weight") != std::string::npos) {
            if (type == InitType::Orthogonal) {
                // For LSTM, initialize each gate separately
                int64_t num_gates = 4; // LSTM has 4 gates
                int64_t gate_size = tensor.size(0) / num_gates;
                
                for (int64_t i = 0; i < num_gates; ++i) {
                    torch::Tensor gate_weight = tensor.slice(0, i * gate_size, (i + 1) * gate_size);
                    OrthogonalInit(gate_weight, gain);
                }
            } else {
                InitializeTensor(tensor, type, Activation::Linear, gain);
            }
        } else if (name.find("bias") != std::string::npos) {
            torch::nn::init::zeros_(tensor);
            
            // Initialize forget gate bias to 1 (common practice)
            int64_t num_gates = 4;
            int64_t gate_size = tensor.size(0) / num_gates;
            int64_t forget_gate_start = gate_size; // Second gate is forget gate
            tensor.slice(0, forget_gate_start, forget_gate_start + gate_size).fill_(1.0f);
        }
    }
}

void WeightInitializer::InitializeGRU(torch::nn::GRU& gru,
                                     InitType type,
                                     float gain) {
    torch::NoGradGuard no_grad;
    
    for (auto& param : gru->named_parameters()) {
        const std::string& name = param.key();
        torch::Tensor& tensor = param.value();
        
        if (name.find("weight") != std::string::npos) {
            if (type == InitType::Orthogonal) {
                // For GRU, initialize each gate separately
                int64_t num_gates = 3; // GRU has 3 gates
                int64_t gate_size = tensor.size(0) / num_gates;
                
                for (int64_t i = 0; i < num_gates; ++i) {
                    torch::Tensor gate_weight = tensor.slice(0, i * gate_size, (i + 1) * gate_size);
                    OrthogonalInit(gate_weight, gain);
                }
            } else {
                InitializeTensor(tensor, type, Activation::Linear, gain);
            }
        } else if (name.find("bias") != std::string::npos) {
            torch::nn::init::zeros_(tensor);
        }
    }
}

float WeightInitializer::CalculateGain(Activation activation, float param) {
    switch (activation) {
        case Activation::Linear:
        case Activation::Sigmoid:
            return 1.0f;
        case Activation::Tanh:
            return 5.0f / 3.0f;
        case Activation::ReLU:
            return std::sqrt(2.0f);
        case Activation::LeakyReLU:
            return std::sqrt(2.0f / (1.0f + param * param));
        case Activation::SELU:
            return 3.0f / 4.0f;
        default:
            return 1.0f;
    }
}

float WeightInitializer::CalculateFanIn(const torch::Tensor& tensor) {
    if (tensor.dim() < 2) {
        return 1.0f;
    } else if (tensor.dim() == 2) {
        return static_cast<float>(tensor.size(1));
    } else {
        // For conv layers: fan_in = input_channels * kernel_size
        float fan_in = static_cast<float>(tensor.size(1));
        for (int64_t i = 2; i < tensor.dim(); ++i) {
            fan_in *= static_cast<float>(tensor.size(i));
        }
        return fan_in;
    }
}

float WeightInitializer::CalculateFanOut(const torch::Tensor& tensor) {
    if (tensor.dim() < 2) {
        return 1.0f;
    } else if (tensor.dim() == 2) {
        return static_cast<float>(tensor.size(0));
    } else {
        // For conv layers: fan_out = output_channels * kernel_size
        float fan_out = static_cast<float>(tensor.size(0));
        for (int64_t i = 2; i < tensor.dim(); ++i) {
            fan_out *= static_cast<float>(tensor.size(i));
        }
        return fan_out;
    }
}

float WeightInitializer::CalculateFanAvg(const torch::Tensor& tensor) {
    return (CalculateFanIn(tensor) + CalculateFanOut(tensor)) / 2.0f;
}

void WeightInitializer::InitializeAttention(torch::nn::Module& attention_module,
                                           float scale) {
    torch::NoGradGuard no_grad;
    
    for (auto& param : attention_module.named_parameters()) {
        const std::string& name = param.key();
        torch::Tensor& tensor = param.value();
        
        if (name.find("weight") != std::string::npos) {
            // Scale down attention weights
            XavierNormal(tensor, scale);
        } else if (name.find("bias") != std::string::npos) {
            torch::nn::init::zeros_(tensor);
        }
    }
}

void WeightInitializer::InitializeBatchNorm(torch::nn::BatchNorm2d& bn,
                                           float weight_value,
                                           float bias_value) {
    torch::NoGradGuard no_grad;
    
    if (bn->weight.defined()) {
        torch::nn::init::constant_(bn->weight, weight_value);
    }
    if (bn->bias.defined()) {
        torch::nn::init::constant_(bn->bias, bias_value);
    }
}

void WeightInitializer::InitializeLayerNorm(torch::nn::LayerNorm& ln,
                                           float weight_value,
                                           float bias_value) {
    torch::NoGradGuard no_grad;
    
    if (ln->weight.defined()) {
        torch::nn::init::constant_(ln->weight, weight_value);
    }
    if (ln->bias.defined()) {
        torch::nn::init::constant_(ln->bias, bias_value);
    }
}

// Private helper functions
void WeightInitializer::XavierUniform(torch::Tensor& tensor, float gain) {
    torch::nn::init::xavier_uniform_(tensor, gain);
}

void WeightInitializer::XavierNormal(torch::Tensor& tensor, float gain) {
    torch::nn::init::xavier_normal_(tensor, gain);
}

void WeightInitializer::KaimingUniform(torch::Tensor& tensor, Activation activation) {
    // Note: PyTorch's kaiming_uniform_ uses its own gain calculation internally
    (void)activation; // Suppress unused parameter warning - may be used in future implementations
    torch::nn::init::kaiming_uniform_(tensor, 0.0f, torch::kFanIn, torch::kReLU);
}

void WeightInitializer::KaimingNormal(torch::Tensor& tensor, Activation activation) {
    // Note: PyTorch's kaiming_normal_ uses its own gain calculation internally
    (void)activation; // Suppress unused parameter warning - may be used in future implementations
    torch::nn::init::kaiming_normal_(tensor, 0.0f, torch::kFanIn, torch::kReLU);
}

void WeightInitializer::OrthogonalInit(torch::Tensor& tensor, float gain) {
    torch::nn::init::orthogonal_(tensor, gain);
}

} // namespace utils
