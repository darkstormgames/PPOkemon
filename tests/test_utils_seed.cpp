#include <iostream>
#include <filesystem>
#include "torch/utils/seed_manager.h"
#include "test_utils.h"

using namespace PPOkemonTest;

void test_seed_manager_basic() {
    // Test global seed setting
    utils::SeedManager::SetGlobalSeed(12345);
    
    // Test reproducibility
    auto seeds1 = utils::SeedManager::GenerateSeeds(42, 5);
    auto seeds2 = utils::SeedManager::GenerateSeeds(42, 5);
    
    bool reproducible = true;
    for (size_t i = 0; i < seeds1.size(); ++i) {
        if (seeds1[i] != seeds2[i]) {
            reproducible = false;
            break;
        }
    }
    
    ASSERT_TRUE(reproducible);
    ASSERT_EQ(utils::SeedManager::GetCurrentSeed(), 12345);
}

void test_seed_manager_random_utilities() {
    utils::SeedManager::SetGlobalSeed(54321);
    
    // Test random utilities
    float rand_float = utils::SeedManager::RandomFloat(0.0f, 1.0f);
    int rand_int = utils::SeedManager::RandomInt(1, 100);
    bool rand_bool = utils::SeedManager::RandomBool(0.7f);
    
    ASSERT_TRUE(rand_float >= 0.0f && rand_float <= 1.0f);
    ASSERT_TRUE(rand_int >= 1 && rand_int <= 100);
    // rand_bool can be either true or false, just test it doesn't crash
    (void)rand_bool; // Suppress unused variable warning
}

void test_seed_manager_deterministic_mode() {
    // Test deterministic mode
    ASSERT_NO_THROW(utils::SeedManager::EnableDeterministicMode());
}

void test_seed_manager_save_load() {
    // Test save/load state
    std::string state_path = "./test_seed_state";
    
    utils::SeedManager::SetGlobalSeed(98765);
    ASSERT_NO_THROW(utils::SeedManager::SaveRandomState(state_path));
    
    utils::SeedManager::SetGlobalSeed(999);
    ASSERT_NO_THROW(utils::SeedManager::LoadRandomState(state_path));
    
    // Cleanup
    std::filesystem::remove(state_path);
    std::filesystem::remove(state_path + ".names");
}

int main() {
    TestSuite suite("PPOkemon Seed Manager Tests");
    
    suite.AddTest("Basic Functionality", test_seed_manager_basic);
    suite.AddTest("Random Utilities", test_seed_manager_random_utilities);
    suite.AddTest("Deterministic Mode", test_seed_manager_deterministic_mode);
    suite.AddTest("Save/Load State", test_seed_manager_save_load);
    
    bool all_passed = suite.RunAll();
    
    int passed, failed, total;
    suite.GetStats(passed, failed, total);
    
    std::cout << "\nFinal Results: " << passed << "/" << total << " tests passed" << std::endl;
    
    return all_passed ? 0 : 1;
}