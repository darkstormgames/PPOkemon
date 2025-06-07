#include <iostream>
#include <torch/torch.h>
#include "torch/utils/lr_scheduler.h"
#include "test_utils.h"

using namespace PPOkemonTest;

void test_lr_scheduler_linear_decay() {
    // Create a simple model and optimizer
    torch::nn::Linear model(10, 1);
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.1));
    
    utils::LearningRateScheduler scheduler(optimizer, utils::SchedulerType::LINEAR_DECAY, 0.1f, 1000);
    scheduler.SetLinearDecay(0.01f);
    
    float initial_lr = scheduler.GetLR();
    ASSERT_NEAR(initial_lr, 0.1f, 0.001f);
    
    scheduler.Step(500);  // Halfway
    float mid_lr = scheduler.GetLR();
    ASSERT_TRUE(mid_lr < initial_lr);
    ASSERT_TRUE(mid_lr > 0.01f);
    
    scheduler.Step(1000); // End
    float final_lr = scheduler.GetLR();
    ASSERT_NEAR(final_lr, 0.01f, 0.001f);
}

void test_lr_scheduler_cosine_annealing() {
    torch::nn::Linear model(10, 1);
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.1));
    
    utils::LearningRateScheduler scheduler(optimizer, utils::SchedulerType::COSINE_ANNEALING, 0.1f, 100);
    scheduler.SetCosineAnnealing(0.001f);
    
    float initial_lr = scheduler.GetLR();
    ASSERT_NEAR(initial_lr, 0.1f, 0.001f);
    
    scheduler.Step(50);  // Halfway
    float mid_lr = scheduler.GetLR();
    ASSERT_TRUE(mid_lr < initial_lr);
    
    scheduler.Step(100); // End
    float final_lr = scheduler.GetLR();
    ASSERT_NEAR(final_lr, 0.001f, 0.001f);
}

void test_lr_scheduler_custom() {
    torch::nn::Linear model(10, 1);
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.1));
    
    utils::LearningRateScheduler scheduler(optimizer, utils::SchedulerType::CUSTOM, 0.1f, 1000);
    scheduler.SetCustomScheduler([](int64_t step, int64_t total_steps, float initial_lr) {
        (void)total_steps; // Suppress unused parameter warning
        // Custom exponential decay
        float decay_rate = 0.95f;
        float decay_steps = 100.0f;
        return initial_lr * std::pow(decay_rate, step / decay_steps);
    });
    
    float initial_lr = scheduler.GetLR();
    ASSERT_NEAR(initial_lr, 0.1f, 0.001f);
    
    scheduler.Step(200);
    float decayed_lr = scheduler.GetLR();
    ASSERT_TRUE(decayed_lr < initial_lr);
    ASSERT_TRUE(decayed_lr > 0.05f); // Should still be reasonable
}

void test_lr_scheduler_step_decay() {
    torch::nn::Linear model(10, 1);
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.1));
    
    utils::LearningRateScheduler scheduler(optimizer, utils::SchedulerType::STEP_DECAY, 0.1f, 1000);
    scheduler.SetStepDecay(0.5f, 100); // Decay by 0.5 every 100 steps
    
    float initial_lr = scheduler.GetLR();
    ASSERT_NEAR(initial_lr, 0.1f, 0.001f);
    
    scheduler.Step(100);
    float first_decay_lr = scheduler.GetLR();
    ASSERT_NEAR(first_decay_lr, 0.05f, 0.001f);
    
    scheduler.Step(200);
    float second_decay_lr = scheduler.GetLR();
    ASSERT_NEAR(second_decay_lr, 0.025f, 0.001f);
}

int main() {
    TestSuite suite("PPOkemon Learning Rate Scheduler Tests");
    
    suite.AddTest("Linear Decay", test_lr_scheduler_linear_decay);
    suite.AddTest("Cosine Annealing", test_lr_scheduler_cosine_annealing);
    suite.AddTest("Custom Scheduler", test_lr_scheduler_custom);
    suite.AddTest("Step Decay", test_lr_scheduler_step_decay);
    
    bool all_passed = suite.RunAll();
    
    int passed, failed, total;
    suite.GetStats(passed, failed, total);
    
    std::cout << "\nFinal Results: " << passed << "/" << total << " tests passed" << std::endl;
    
    return all_passed ? 0 : 1;
}