#include "torch/envs/env_base.h"
#include "torch/utils/running_mean_std.h"
#include <torch/torch.h>
#include <memory>
#include <string>
#include <cmath>
#include <cstring>

BaseEnv::BaseEnv(const unsigned int seed)
    : AbstractEnv(seed)
    , obs_normalization_enabled_(false)
    , reward_normalization_enabled_(false)
    , obs_epsilon_(1e-8f)
    , reward_epsilon_(1e-8f)
{
    // Initialize running mean/std objects when needed
}

BaseEnv::~BaseEnv()
{
    // unique_ptr handles cleanup automatically
}

void BaseEnv::SetObservationNormalization(bool enable, float epsilon)
{
    obs_normalization_enabled_ = enable;
    obs_epsilon_ = epsilon;
    
    if (enable && !obs_running_mean_std_)
    {
        // Initialize with observation shape
        auto obs_shape = GetObservationShape();
        obs_running_mean_std_ = std::make_unique<RunningMeanStd>(obs_shape, epsilon);
        
        // Initialize raw observation buffer
        raw_obs_buffer_.resize(GetObservationSize());
    }
}

void BaseEnv::SetRewardNormalization(bool enable, float epsilon)
{
    reward_normalization_enabled_ = enable;
    reward_epsilon_ = epsilon;
    
    if (enable && !reward_running_mean_std_)
    {
        // Initialize with scalar shape for rewards
        reward_running_mean_std_ = std::make_unique<RunningMeanStd>(torch::IntArrayRef{}, epsilon);
    }
}

void BaseEnv::GetNormalizedObsData(float* buffer) const
{
    if (!obs_normalization_enabled_ || !obs_running_mean_std_)
    {
        // No normalization, just get raw observation
        GetObsData(buffer);
        return;
    }
    
    // Get raw observation data
    GetObsData(raw_obs_buffer_.data());
    
    // Convert to torch tensor
    const int64_t obs_size = GetObservationSize();
    torch::Tensor obs_tensor = torch::from_blob(
        const_cast<float*>(raw_obs_buffer_.data()), 
        GetObservationShape(), 
        torch::kFloat32
    );
    
    // Normalize using running mean/std
    torch::NoGradGuard no_grad;
    torch::Tensor mean = obs_running_mean_std_->GetMean();
    torch::Tensor var = obs_running_mean_std_->GetVar();
    torch::Tensor normalized = (obs_tensor - mean) / torch::sqrt(var + obs_epsilon_);
    
    // Copy normalized data back to buffer
    std::memcpy(buffer, normalized.data_ptr<float>(), obs_size * sizeof(float));
}

void BaseEnv::UpdateNormalizationStats(const torch::Tensor& obs_batch, const torch::Tensor& reward_batch)
{
    torch::NoGradGuard no_grad;
    
    if (obs_normalization_enabled_ && obs_running_mean_std_)
    {
        obs_running_mean_std_->Update(obs_batch);
    }
    
    if (reward_normalization_enabled_ && reward_running_mean_std_)
    {
        // Reshape rewards to ensure proper shape for update
        torch::Tensor rewards_reshaped = reward_batch.view({-1, 1});
        reward_running_mean_std_->Update(rewards_reshaped);
    }
}

float BaseEnv::NormalizeReward(float reward) const
{
    if (!reward_normalization_enabled_ || !reward_running_mean_std_)
    {
        return reward;
    }
    
    torch::NoGradGuard no_grad;
    float mean = reward_running_mean_std_->GetMean().item<float>();
    float var = reward_running_mean_std_->GetVar().item<float>();
    
    return (reward - mean) / std::sqrt(var + reward_epsilon_);
}

float BaseEnv::DenormalizeReward(float normalized_reward) const
{
    if (!reward_normalization_enabled_ || !reward_running_mean_std_)
    {
        return normalized_reward;
    }
    
    torch::NoGradGuard no_grad;
    float mean = reward_running_mean_std_->GetMean().item<float>();
    float var = reward_running_mean_std_->GetVar().item<float>();
    
    return normalized_reward * std::sqrt(var + reward_epsilon_) + mean;
}

void BaseEnv::SaveNormalizationStats(const std::string& obs_path, const std::string& reward_path) const
{
    if (obs_normalization_enabled_ && obs_running_mean_std_)
    {
        obs_running_mean_std_->Save(obs_path);
    }
    
    if (reward_normalization_enabled_ && reward_running_mean_std_)
    {
        reward_running_mean_std_->Save(reward_path);
    }
}

void BaseEnv::LoadNormalizationStats(const std::string& obs_path, const std::string& reward_path)
{
    if (obs_normalization_enabled_ && obs_running_mean_std_)
    {
        obs_running_mean_std_->Load(obs_path);
    }
    
    if (reward_normalization_enabled_ && reward_running_mean_std_)
    {
        reward_running_mean_std_->Load(reward_path);
    }
}