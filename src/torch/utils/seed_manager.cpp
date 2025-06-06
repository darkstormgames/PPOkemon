#include "torch/utils/seed_manager.h"
#include <torch/torch.h>
#include <fstream>
#include <chrono>
#include <sstream>
#include <cstdlib>
#include <iostream>

namespace utils {

// Static member definitions
unsigned int SeedManager::current_seed_ = 0;
std::mt19937 SeedManager::generator_(std::chrono::steady_clock::now().time_since_epoch().count());
bool SeedManager::deterministic_mode_ = false;

void SeedManager::SetGlobalSeed(unsigned int seed) {
    current_seed_ = seed;
    SetTorchSeed(seed);
    SetCudaSeed(seed);
    SetStdSeed(seed);
    
    if (deterministic_mode_) {
        EnableDeterministicMode();
    }
}

void SeedManager::SetTorchSeed(unsigned int seed) {
    torch::manual_seed(seed);
}

void SeedManager::SetCudaSeed(unsigned int seed) {
    if (torch::cuda::is_available()) {
        torch::cuda::manual_seed(seed);
        torch::cuda::manual_seed_all(seed);
    }
}

void SeedManager::SetStdSeed(unsigned int seed) {
    generator_.seed(seed);
}

std::vector<unsigned int> SeedManager::GenerateSeeds(unsigned int base_seed, size_t count) {
    std::mt19937 temp_gen(base_seed);
    std::vector<unsigned int> seeds;
    seeds.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        seeds.push_back(temp_gen());
    }
    
    return seeds;
}

unsigned int SeedManager::GetCurrentSeed() {
    return current_seed_;
}

void SeedManager::SaveRandomState(const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    
    // Save current seed
    file.write(reinterpret_cast<const char*>(&current_seed_), sizeof(current_seed_));
    
    // Save generator state
    std::ostringstream gen_state;
    gen_state << generator_;
    std::string state_str = gen_state.str();
    size_t state_size = state_str.size();
    file.write(reinterpret_cast<const char*>(&state_size), sizeof(state_size));
    file.write(state_str.c_str(), state_size);
    
    // Save deterministic mode flag
    file.write(reinterpret_cast<const char*>(&deterministic_mode_), sizeof(deterministic_mode_));
    
    // Save PyTorch RNG state (simplified approach)
    try {
        // For compatibility, we'll just store the current seed
        int64_t torch_seed = current_seed_;
        file.write(reinterpret_cast<const char*>(&torch_seed), sizeof(torch_seed));
        std::cout << "PyTorch seed saved: " << torch_seed << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not save PyTorch RNG state: " << e.what() << std::endl;
        int64_t torch_seed = 0;
        file.write(reinterpret_cast<const char*>(&torch_seed), sizeof(torch_seed));
    }
}

void SeedManager::LoadRandomState(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }
    
    // Load current seed
    file.read(reinterpret_cast<char*>(&current_seed_), sizeof(current_seed_));
    
    // Load generator state
    size_t state_size;
    file.read(reinterpret_cast<char*>(&state_size), sizeof(state_size));
    std::string state_str(state_size, '\0');
    file.read(&state_str[0], state_size);
    std::istringstream gen_state(state_str);
    gen_state >> generator_;
    
    // Load deterministic mode flag
    file.read(reinterpret_cast<char*>(&deterministic_mode_), sizeof(deterministic_mode_));
    
    // Load PyTorch RNG state (simplified approach)
    try {
        int64_t torch_seed;
        file.read(reinterpret_cast<char*>(&torch_seed), sizeof(torch_seed));
        if (torch_seed != 0) {
            torch::manual_seed(torch_seed);
            if (torch::cuda::is_available()) {
                torch::cuda::manual_seed_all(torch_seed);
            }
            std::cout << "PyTorch seed restored: " << torch_seed << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not load PyTorch RNG state: " << e.what() << std::endl;
    }
    
    if (deterministic_mode_) {
        EnableDeterministicMode();
    }
}

void SeedManager::EnableDeterministicMode() {
    deterministic_mode_ = true;
    
    // Enable deterministic algorithms in PyTorch
    try {
        torch::manual_seed(current_seed_);
        if (torch::cuda::is_available()) {
            torch::cuda::manual_seed_all(current_seed_);
        }
        // Note: torch::set_deterministic and torch::set_warn_always may not be available in all PyTorch versions
        std::cout << "Deterministic mode enabled (manual seed set)" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not fully enable deterministic mode: " << e.what() << std::endl;
    }
    
    // Set environment variables for better determinism
    setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 1);
}

void SeedManager::DisableDeterministicMode() {
    deterministic_mode_ = false;
    std::cout << "Deterministic mode disabled" << std::endl;
}

std::mt19937& SeedManager::GetGenerator() {
    return generator_;
}

float SeedManager::RandomFloat(float min, float max) {
    std::uniform_real_distribution<float> dist(min, max);
    return dist(generator_);
}

int SeedManager::RandomInt(int min, int max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(generator_);
}

bool SeedManager::RandomBool(float probability) {
    std::bernoulli_distribution dist(probability);
    return dist(generator_);
}

} // namespace utils
