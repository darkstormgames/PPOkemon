#include <iostream>
#include <torch/torch.h>

int main() {
    std::cout << "PPOkemon Deep Reinforcement Learning Framework" << std::endl;
    std::cout << "===============================================" << std::endl;
    
    // Check torch availability
    std::cout << "PyTorch Version: " << TORCH_VERSION_MAJOR << "." 
              << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH << std::endl;
    
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available with " << torch::cuda::device_count() << " device(s)" << std::endl;
    } else {
        std::cout << "CUDA is not available, using CPU" << std::endl;
    }
    
    std::cout << "\nFramework initialized successfully!" << std::endl;
    std::cout << "Use the test executables to verify implementations." << std::endl;
    
    return 0;
}