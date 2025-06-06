#pragma once

#include <torch/torch.h>
#include <memory>
#include <functional>

namespace utils {

enum class SchedulerType {
    CONSTANT,
    LINEAR_DECAY,
    EXPONENTIAL_DECAY,
    COSINE_ANNEALING,
    STEP_DECAY,
    POLYNOMIAL_DECAY,
    CUSTOM
};

class LearningRateScheduler {
public:
    LearningRateScheduler(torch::optim::Optimizer& optimizer, SchedulerType type, 
                         float initial_lr, int64_t total_steps);
    ~LearningRateScheduler() = default;
    
    // Update learning rate based on current step
    void Step(int64_t current_step);
    
    // Get current learning rate
    float GetLR() const;
    
    // Reset scheduler
    void Reset();
    
    // Set custom scheduler function
    void SetCustomScheduler(std::function<float(int64_t, int64_t, float)> scheduler_fn);
    
    // Configuration methods for different scheduler types
    void SetLinearDecay(float end_lr = 0.0f);
    void SetExponentialDecay(float decay_rate = 0.95f);
    void SetCosineAnnealing(float min_lr = 0.0f);
    void SetStepDecay(float decay_rate = 0.1f, int64_t step_size = 1000);
    void SetPolynomialDecay(float end_lr = 0.0f, float power = 1.0f);

private:
    torch::optim::Optimizer& optimizer_;
    SchedulerType type_;
    float initial_lr_;
    float current_lr_;
    int64_t total_steps_;
    int64_t current_step_;
    
    // Scheduler-specific parameters
    float end_lr_;
    float decay_rate_;
    float min_lr_;
    int64_t step_size_;
    float power_;
    
    // Custom scheduler function
    std::function<float(int64_t, int64_t, float)> custom_scheduler_;
    
    // Scheduler implementations
    float ComputeLinearDecay(int64_t step);
    float ComputeExponentialDecay(int64_t step);
    float ComputeCosineAnnealing(int64_t step);
    float ComputeStepDecay(int64_t step);
    float ComputePolynomialDecay(int64_t step);
    
    // Update optimizer learning rate
    void UpdateOptimizerLR(float new_lr);
};

} // namespace utils
