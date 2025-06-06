#include "include/torch/networks/network_base.h"
#include "include/torch/networks/mlp.h"
#include "include/torch/networks/cnn.h"
#include "include/torch/networks/a2c.h"
#include "include/torch/networks/a3c.h"
#include <iostream>

int main() {
    try {
        std::cout << "Testing Network Implementations...\n";
        
        // Test MLP Network
        std::cout << "1. Testing MLP Network...\n";
        auto mlp = networks::MLP(10, 64, 32);
        auto input = torch::randn({1, 10});
        auto output = mlp->forward(input);
        std::cout << "   MLP Output shape: [" << output.size(0) << ", " << output.size(1) << "]\n";
        
        // Test CNN Network
        std::cout << "2. Testing CNN Network...\n";
        auto cnn = networks::CNNBody(1, 160, 144);  // Gameboy screen
        auto img_input = torch::randn({1, 1, 160, 144});
        auto cnn_output = cnn->forward(img_input);
        std::cout << "   CNN Output shape: [" << cnn_output.size(0) << ", " << cnn_output.size(1) << "]\n";
        
        // Test A2C Network with MLP
        std::cout << "3. Testing A2C Network with MLP...\n";
        auto a2c_mlp = networks::A2C(networks::MLPTag{}, 10, 64, 4);  // 4 actions
        auto a2c_output = a2c_mlp->forward(input);
        std::cout << "   A2C-MLP Action shape: [" << a2c_output.size(0) << ", " << a2c_output.size(1) << "]\n";
        
        // Test A2C Network with CNN
        std::cout << "4. Testing A2C Network with CNN...\n";
        auto a2c_cnn = networks::A2C(networks::CNNTag{}, 1, 160, 144, 4);  // 4 actions
        auto a2c_cnn_output = a2c_cnn->forward(img_input);
        std::cout << "   A2C-CNN Action shape: [" << a2c_cnn_output.size(0) << ", " << a2c_cnn_output.size(1) << "]\n";
        
        // Test A3C Network with MLP
        std::cout << "5. Testing A3C Network with MLP...\n";
        auto a3c_mlp = networks::A3C(networks::MLPTag{}, 10, 64, 4);  // 4 actions
        auto a3c_output = a3c_mlp->forward(input);
        std::cout << "   A3C-MLP Action shape: [" << a3c_output.size(0) << ", " << a3c_output.size(1) << "]\n";
        
        // Test Actor-Critic methods
        std::cout << "6. Testing Actor-Critic methods...\n";
        auto [actions, values] = a2c_mlp->ForwardActorCritic(input);
        std::cout << "   Actions shape: [" << actions.size(0) << ", " << actions.size(1) << "]\n";
        std::cout << "   Values shape: [" << values.size(0) << ", " << values.size(1) << "]\n";
        
        // Test orthogonal initialization
        std::cout << "7. Testing orthogonal initialization...\n";
        mlp->InitOrtho(1.0f);
        a2c_mlp->InitOrtho(1.0f);
        a3c_mlp->InitOrtho(1.0f);
        std::cout << "   Orthogonal initialization completed successfully\n";
        
        std::cout << "\nAll tests passed successfully!\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
}
