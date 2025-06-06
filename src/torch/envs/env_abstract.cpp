#include "torch/envs/env_abstract.h"
#include <thread>
#include <chrono>

AbstractEnv::AbstractEnv(const unsigned int seed)
{
    if (seed == 0)
    {
        std::random_device rd;
        random_engine = std::mt19937(rd());
    }
    else
    {
        random_engine = std::mt19937(seed);
    }
    current_episode_length = 0;
    current_episode_reward = 0.0f;
}

AbstractEnv::~AbstractEnv()
{

}

void AbstractEnv::SetSeed(unsigned int seed)
{
    if (seed == 0)
    {
        std::random_device rd;
        random_engine = std::mt19937(rd());
    }
    else
    {
        random_engine = std::mt19937(seed);
    }
}

void AbstractEnv::Reset()
{
    ResetImpl();
    current_episode_length = 0;
    current_episode_reward = 0.0f;
}

StepResultRaw AbstractEnv::Step(const float* action_data, int64_t action_size)
{
    current_episode_length += 1;
    StepResultRaw result = StepImpl(action_data, action_size);
    current_episode_reward += result.reward;

    if (result.done)
    {
        result.tot_reward = current_episode_reward;
        result.tot_steps = current_episode_length;
        // Reset for next episode
        Reset();
    }
    return result;
}

void AbstractEnv::Render(const uint64_t wait_ms)
{
    RenderImpl();
    if (wait_ms > 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
    }
}