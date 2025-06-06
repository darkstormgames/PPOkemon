#include "torch/envs/env_vectorized.h"
#include <functional>

VectorizedEnv::VectorizedEnv(std::function<std::unique_ptr<AbstractEnv>()> env_factory, 
                             int num_envs, 
                             const torch::Device& device)
    : num_envs_(num_envs), device_(device)
{
    envs_.reserve(num_envs);
    obs_buffers_.reserve(num_envs);
    action_buffers_.reserve(num_envs);
    
    // Create all environment instances
    for (int i = 0; i < num_envs; ++i)
    {
        envs_.push_back(env_factory());
        if (!envs_[i])
        {
            throw std::runtime_error("Environment factory returned null pointer for environment " + std::to_string(i));
        }
        
        // Pre-allocate buffers for this environment
        obs_buffers_.emplace_back(envs_[i]->GetObservationSize());
        action_buffers_.emplace_back(envs_[i]->GetActionSize());
    }
}

VectorizedEnv::~VectorizedEnv()
{
    // envs_ will be automatically destroyed due to unique_ptr
}

int64_t VectorizedEnv::GetObservationSize() const
{
    if (envs_.empty()) return 0;
    return envs_[0]->GetObservationSize();
}

std::vector<int64_t> VectorizedEnv::GetObservationShape() const
{
    if (envs_.empty()) return {};
    return envs_[0]->GetObservationShape();
}

int64_t VectorizedEnv::GetActionSize() const
{
    if (envs_.empty()) return 0;
    return envs_[0]->GetActionSize();
}

torch::Tensor VectorizedEnv::Reset()
{
    const int64_t obs_size = GetObservationSize();
    const auto obs_shape = GetObservationShape();
    
    // Create output tensor with appropriate shape
    std::vector<int64_t> output_shape = {num_envs_};
    output_shape.insert(output_shape.end(), obs_shape.begin(), obs_shape.end());
    
    torch::Tensor observations = torch::zeros(output_shape, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    
    // Reset all environments and collect observations
    for (int i = 0; i < num_envs_; ++i)
    {
        envs_[i]->Reset();
        envs_[i]->GetObsData(obs_buffers_[i].data());
        
        // Copy observation data to tensor
        torch::Tensor obs_tensor = torch::from_blob(obs_buffers_[i].data(), {obs_size}, torch::kFloat32);
        observations[i] = obs_tensor.to(device_);
    }
    
    return observations;
}

std::tuple<torch::Tensor, VectorizedStepResult> VectorizedEnv::Step(const torch::Tensor& actions)
{
    const int64_t obs_size = GetObservationSize();
    const int64_t action_size = GetActionSize();
    const auto obs_shape = GetObservationShape();
    
    // Verify action tensor shape
    if (actions.size(0) != num_envs_ || actions.size(1) != action_size)
    {
        throw std::runtime_error("Actions tensor has wrong shape. Expected [" + 
                                std::to_string(num_envs_) + ", " + std::to_string(action_size) + 
                                "], got [" + std::to_string(actions.size(0)) + ", " + 
                                std::to_string(actions.size(1)) + "]");
    }
    
    // Create output tensors
    std::vector<int64_t> output_shape = {num_envs_};
    output_shape.insert(output_shape.end(), obs_shape.begin(), obs_shape.end());
    
    torch::Tensor observations = torch::zeros(output_shape, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    torch::Tensor rewards = torch::zeros({num_envs_}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    torch::Tensor dones = torch::zeros({num_envs_}, torch::TensorOptions().dtype(torch::kBool).device(device_));
    torch::Tensor tot_rewards = torch::zeros({num_envs_}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    torch::Tensor tot_steps = torch::zeros({num_envs_}, torch::TensorOptions().dtype(torch::kInt64).device(device_));
    
    // Convert actions to CPU for environment stepping
    torch::Tensor actions_cpu = actions.to(torch::kCPU);
    
    // Step all environments
    for (int i = 0; i < num_envs_; ++i)
    {
        // Copy action data to buffer
        std::memcpy(action_buffers_[i].data(), actions_cpu[i].data_ptr<float>(), action_size * sizeof(float));
        
        // Step the environment
        StepResultRaw result = envs_[i]->Step(action_buffers_[i].data(), action_size);
        
        // Get new observation
        envs_[i]->GetObsData(obs_buffers_[i].data());
        
        // Copy data to output tensors
        torch::Tensor obs_tensor = torch::from_blob(obs_buffers_[i].data(), {obs_size}, torch::kFloat32);
        observations[i] = obs_tensor.to(device_);
        
        rewards[i] = result.reward;
        dones[i] = result.done;
        
        if (result.done)
        {
            tot_rewards[i] = result.tot_reward;
            tot_steps[i] = static_cast<int64_t>(result.tot_steps);
        }
    }
    
    VectorizedStepResult step_result;
    step_result.rewards = rewards;
    step_result.dones = dones;
    step_result.tot_rewards = tot_rewards;
    step_result.tot_steps = tot_steps;
    
    return std::make_tuple(observations, step_result);
}

void VectorizedEnv::Render(uint64_t wait_ms)
{
    for (int i = 0; i < num_envs_; ++i)
    {
        envs_[i]->Render(wait_ms);
    }
}

void VectorizedEnv::SetSeed(unsigned int seed)
{
    for (int i = 0; i < num_envs_; ++i)
    {
        envs_[i]->SetSeed(seed + i);
    }
}