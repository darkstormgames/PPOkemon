#include <iostream>
#include <torch/torch.h>

// Include the new utility headers
#include "torch/utils/ema.h"
#include "torch/utils/advantage_estimator.h"
#include "torch/utils/weight_initializer.h"
#include "torch/utils/noise.h"

void test_ema() {
    std::cout << "\n=== Testing Exponential Moving Average ===" << std::endl;
    
    // Create a simple linear layer
    torch::nn::Linear linear(torch::nn::LinearOptions(10, 5));
    
    // Initialize EMA
    utils::ExponentialMovingAverage ema(0.999f);
    ema.RegisterModel(*linear);
    
    // Simulate training updates
    for (int i = 0; i < 10; ++i) {
        // Modify weights (simulate gradient update) - use non-in-place operation
        {
            torch::NoGradGuard no_grad;
            auto noise = torch::randn_like(linear->weight) * 0.01f;
            linear->weight.copy_(linear->weight + noise);
        }
        
        // Update EMA
        ema.Update(*linear);
    }
    
    std::cout << "EMA parameter count: " << ema.GetParameterCount() << std::endl;
    std::cout << "EMA test passed!" << std::endl;
}

void test_advantage_estimator() {
    std::cout << "\n=== Testing Advantage Estimator ===" << std::endl;
    
    // Create sample trajectory data
    int seq_len = 100;
    torch::Tensor rewards = torch::randn({seq_len}) * 0.1f;
    torch::Tensor values = torch::randn({seq_len});
    torch::Tensor dones = torch::zeros({seq_len});
    
    // Set some episodes to be done
    dones[25] = 1.0f;
    dones[60] = 1.0f;
    dones[99] = 1.0f;
    
    // Test GAE
    torch::Tensor gae_advantages = utils::AdvantageEstimator::ComputeGAE(
        rewards, values, dones, 0.99f, 0.95f);
    
    // Test TD advantage
    torch::Tensor td_advantages = utils::AdvantageEstimator::ComputeTDAdvantage(
        rewards, values, dones, 0.99f);
    
    // Test normalization
    torch::Tensor normalized_advantages = utils::AdvantageEstimator::NormalizeAdvantages(gae_advantages);
    
    std::cout << "GAE advantages shape: [" << gae_advantages.size(0) << "]" << std::endl;
    std::cout << "GAE mean: " << gae_advantages.mean().item<float>() << std::endl;
    std::cout << "Normalized advantages std: " << normalized_advantages.std().item<float>() << std::endl;
    std::cout << "Advantage estimator test passed!" << std::endl;
}

void test_weight_initializer() {
    std::cout << "\n=== Testing Weight Initializer ===" << std::endl;
    
    // Create a simple network
    torch::nn::Sequential network(
        torch::nn::Linear(64, 128),
        torch::nn::ReLU(),
        torch::nn::Linear(128, 64),
        torch::nn::ReLU(),
        torch::nn::Linear(64, 10)
    );
    
    // Test orthogonal initialization
    utils::WeightInitializer::InitializeModule(*network, 
        utils::WeightInitializer::InitType::Orthogonal,
        utils::WeightInitializer::Activation::ReLU);
    
    // Test policy head initialization
    torch::nn::Linear policy_head(torch::nn::LinearOptions(64, 4));
    utils::WeightInitializer::InitializePolicyHead(policy_head, 0.01f);
    
    std::cout << "Network initialized with orthogonal weights" << std::endl;
    std::cout << "Policy head weight std: " << policy_head->weight.std().item<float>() << std::endl;
    std::cout << "Weight initializer test passed!" << std::endl;
}

void test_noise_generators() {
    std::cout << "\n=== Testing Noise Generators ===" << std::endl;
    
    // Test Gaussian noise
    utils::GaussianNoise gaussian_noise(0.0f, 0.1f);
    torch::Tensor gaussian_sample = gaussian_noise.Sample({10, 5});
    
    // Test Ornstein-Uhlenbeck noise
    utils::OrnsteinUhlenbeckNoise ou_noise(0.15f, 0.2f);
    torch::Tensor ou_sample1 = ou_noise.Sample({5});
    torch::Tensor ou_sample2 = ou_noise.Sample({5}); // Should be correlated
    
    // Test epsilon-greedy
    utils::EpsilonGreedyNoise eps_greedy(0.1f);
    int random_actions = 0;
    for (int i = 0; i < 1000; ++i) {
        if (eps_greedy.ShouldExplore()) {
            random_actions++;
        }
    }
    
    // Test noise manager
    utils::NoiseManager noise_manager;
    noise_manager.RegisterNoise("gaussian", std::make_unique<utils::GaussianNoise>(0.0f, 0.1f));
    
    torch::Tensor test_tensor = torch::ones({5, 5});
    torch::Tensor noisy_tensor = noise_manager.ApplyNoise("gaussian", test_tensor);
    
    std::cout << "Gaussian noise sample shape: [" << gaussian_sample.size(0) << ", " << gaussian_sample.size(1) << "]" << std::endl;
    std::cout << "OU noise correlation (should be high): " << torch::corrcoef(torch::stack({ou_sample1, ou_sample2}))[0][1].item<float>() << std::endl;
    std::cout << "Epsilon-greedy exploration rate: " << (float)random_actions / 1000.0f << " (should be ~0.1)" << std::endl;
    std::cout << "Noise generators test passed!" << std::endl;
}

int main() {
    std::cout << "Testing PPOkemon ML Utilities" << std::endl;
    std::cout << "============================" << std::endl;
    
    try {
        test_ema();
        test_advantage_estimator();
        test_weight_initializer();
        test_noise_generators();
        
        std::cout << "\n🎉 All utility tests passed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
