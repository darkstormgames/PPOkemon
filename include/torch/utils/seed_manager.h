#pragma once

#include <torch/torch.h>
#include <random>
#include <vector>

namespace utils {

class SeedManager {
public:
    // Set global seed for all random number generators
    static void SetGlobalSeed(unsigned int seed);
    
    // Set seed for specific components
    static void SetTorchSeed(unsigned int seed);
    static void SetCudaSeed(unsigned int seed);
    static void SetStdSeed(unsigned int seed);
    
    // Generate reproducible seeds for different components
    static std::vector<unsigned int> GenerateSeeds(unsigned int base_seed, size_t count);
    
    // Get current seed state
    static unsigned int GetCurrentSeed();
    
    // Save/restore random state
    static void SaveRandomState(const std::string& path);
    static void LoadRandomState(const std::string& path);
    
    // Deterministic operations
    static void EnableDeterministicMode();
    static void DisableDeterministicMode();
    
    // Random utilities
    static std::mt19937& GetGenerator();
    static float RandomFloat(float min = 0.0f, float max = 1.0f);
    static int RandomInt(int min, int max);
    static bool RandomBool(float probability = 0.5f);

private:
    static unsigned int current_seed_;
    static std::mt19937 generator_;
    static bool deterministic_mode_;
};

} // namespace utils
