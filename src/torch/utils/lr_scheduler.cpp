#include "torch/utils/lr_scheduler.h"
#include <cmath>
#include <iostream>

namespace utils {

LearningRateScheduler::LearningRateScheduler(torch::optim::Optimizer& optimizer, 
                                           SchedulerType type, 
                                           float initial_lr, 
                                           int64_t total_steps)
    : optimizer_(optimizer), type_(type), initial_lr_(initial_lr), 
      current_lr_(initial_lr), total_steps_(total_steps), current_step_(0),
      end_lr_(0.0f), decay_rate_(0.95f), min_lr_(0.0f), 
      step_size_(1000), power_(1.0f) {
    
    // Set initial learning rate in optimizer
    UpdateOptimizerLR(initial_lr_);
}

void LearningRateScheduler::Step(int64_t current_step) {
    current_step_ = current_step;
    
    float new_lr = initial_lr_;
    
    switch (type_) {
        case SchedulerType::CONSTANT:
            new_lr = initial_lr_;
            break;
        case SchedulerType::LINEAR_DECAY:
            new_lr = ComputeLinearDecay(current_step);
            break;
        case SchedulerType::EXPONENTIAL_DECAY:
            new_lr = ComputeExponentialDecay(current_step);
            break;
        case SchedulerType::COSINE_ANNEALING:
            new_lr = ComputeCosineAnnealing(current_step);
            break;
        case SchedulerType::STEP_DECAY:
            new_lr = ComputeStepDecay(current_step);
            break;
        case SchedulerType::POLYNOMIAL_DECAY:
            new_lr = ComputePolynomialDecay(current_step);
            break;
        case SchedulerType::CUSTOM:
            if (custom_scheduler_) {
                new_lr = custom_scheduler_(current_step, total_steps_, initial_lr_);
            } else {
                new_lr = initial_lr_;
            }
            break;
    }
    
    current_lr_ = new_lr;
    UpdateOptimizerLR(new_lr);
}

float LearningRateScheduler::GetLR() const {
    return current_lr_;
}

void LearningRateScheduler::Reset() {
    current_step_ = 0;
    current_lr_ = initial_lr_;
    UpdateOptimizerLR(initial_lr_);
}

void LearningRateScheduler::SetCustomScheduler(std::function<float(int64_t, int64_t, float)> scheduler_fn) {
    custom_scheduler_ = scheduler_fn;
    type_ = SchedulerType::CUSTOM;
}

void LearningRateScheduler::SetLinearDecay(float end_lr) {
    end_lr_ = end_lr;
    type_ = SchedulerType::LINEAR_DECAY;
}

void LearningRateScheduler::SetExponentialDecay(float decay_rate) {
    decay_rate_ = decay_rate;
    type_ = SchedulerType::EXPONENTIAL_DECAY;
}

void LearningRateScheduler::SetCosineAnnealing(float min_lr) {
    min_lr_ = min_lr;
    type_ = SchedulerType::COSINE_ANNEALING;
}

void LearningRateScheduler::SetStepDecay(float decay_rate, int64_t step_size) {
    decay_rate_ = decay_rate;
    step_size_ = step_size;
    type_ = SchedulerType::STEP_DECAY;
}

void LearningRateScheduler::SetPolynomialDecay(float end_lr, float power) {
    end_lr_ = end_lr;
    power_ = power;
    type_ = SchedulerType::POLYNOMIAL_DECAY;
}

float LearningRateScheduler::ComputeLinearDecay(int64_t step) {
    if (step >= total_steps_) {
        return end_lr_;
    }
    
    float progress = static_cast<float>(step) / static_cast<float>(total_steps_);
    return initial_lr_ + (end_lr_ - initial_lr_) * progress;
}

float LearningRateScheduler::ComputeExponentialDecay(int64_t step) {
    return initial_lr_ * std::pow(decay_rate_, static_cast<float>(step));
}

float LearningRateScheduler::ComputeCosineAnnealing(int64_t step) {
    if (step >= total_steps_) {
        return min_lr_;
    }
    
    float progress = static_cast<float>(step) / static_cast<float>(total_steps_);
    float cosine_factor = 0.5f * (1.0f + std::cos(M_PI * progress));
    return min_lr_ + (initial_lr_ - min_lr_) * cosine_factor;
}

float LearningRateScheduler::ComputeStepDecay(int64_t step) {
    int64_t num_decays = step / step_size_;
    return initial_lr_ * std::pow(decay_rate_, static_cast<float>(num_decays));
}

float LearningRateScheduler::ComputePolynomialDecay(int64_t step) {
    if (step >= total_steps_) {
        return end_lr_;
    }
    
    float progress = static_cast<float>(step) / static_cast<float>(total_steps_);
    float decay_factor = std::pow(1.0f - progress, power_);
    return (initial_lr_ - end_lr_) * decay_factor + end_lr_;
}

void LearningRateScheduler::UpdateOptimizerLR(float new_lr) {
    // Update learning rate for all parameter groups in the optimizer
    for (auto& group : optimizer_.param_groups()) {
        group.options().set_lr(new_lr);
    }
}

} // namespace utils
