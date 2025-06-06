#pragma once

#include <random>
#include <stdint.h>
#include <vector>
#include <torch/torch.h>

struct StepResultRaw
{
    float reward;
    bool done;  // Simple boolean instead of TerminalState enum
    float tot_reward;
    uint64_t tot_steps;
};

class AbstractEnv
{
public:
    AbstractEnv(const unsigned int seed = 0);
    virtual ~AbstractEnv();

    /// @brief Get the observation size
    virtual int64_t GetObservationSize() const = 0;
    
    /// @brief Get the observation shape (for CNN policies)
    virtual std::vector<int64_t> GetObservationShape() const { return {GetObservationSize()}; }
    
    /// @brief Get the action size
    virtual int64_t GetActionSize() const = 0;
    
    /// @brief Get observation data into a buffer
    virtual void GetObsData(float* buffer) const = 0;

    /// @brief Reset the environment
    void Reset();
    
    /// @brief Step the environment
    StepResultRaw Step(const float* action_data, int64_t action_size);
    
    /// @brief Render the environment
    void Render(const uint64_t wait_ms = 0);
    
    /// @brief Set the seed for the random number generator
    void SetSeed(unsigned int seed);

protected:
    /// @brief Implementation of reset to be provided by derived classes
    virtual void ResetImpl() = 0;
    
    /// @brief Implementation of step to be provided by derived classes
    virtual StepResultRaw StepImpl(const float* action_data, int64_t action_size) = 0;
    
    /// @brief Implementation of render to be provided by derived classes
    virtual void RenderImpl() = 0;

protected:
    std::mt19937 random_engine;
    uint64_t current_episode_length;
    float current_episode_reward;
};
