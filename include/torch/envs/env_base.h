#pragma once

#include "torch/envs/env_abstract.h"
#include "torch/utils/running_mean_std.h"
#include <torch/torch.h>
#include <memory>
#include <vector>
#include <string>

/// @brief Base environment class that provides common functionality
/// This class extends AbstractEnv with normalization and utility features
class BaseEnv : public AbstractEnv
{
public:
    BaseEnv(const unsigned int seed = 0);
    virtual ~BaseEnv();

    /// @brief Enable/disable observation normalization
    /// @param enable Whether to enable normalization
    /// @param epsilon Small value added to variance for numerical stability
    void SetObservationNormalization(bool enable, float epsilon = 1e-8f);
    
    /// @brief Enable/disable reward normalization
    /// @param enable Whether to enable normalization
    /// @param epsilon Small value added to variance for numerical stability
    void SetRewardNormalization(bool enable, float epsilon = 1e-8f);
    
    /// @brief Get normalized observation data
    /// @param buffer Buffer to fill with normalized observation data
    void GetNormalizedObsData(float* buffer) const;
    
    /// @brief Update normalization statistics (call this during training)
    /// @param obs_batch Batch of observations [batch_size, obs_size]
    /// @param reward_batch Batch of rewards [batch_size]
    void UpdateNormalizationStats(const torch::Tensor& obs_batch, const torch::Tensor& reward_batch);
    
    /// @brief Save normalization statistics to file
    /// @param obs_path Path to save observation normalization stats
    /// @param reward_path Path to save reward normalization stats
    void SaveNormalizationStats(const std::string& obs_path, const std::string& reward_path) const;
    
    /// @brief Load normalization statistics from file
    /// @param obs_path Path to load observation normalization stats
    /// @param reward_path Path to load reward normalization stats
    void LoadNormalizationStats(const std::string& obs_path, const std::string& reward_path);

protected:
    /// @brief Normalize a reward value
    /// @param reward Raw reward
    /// @return Normalized reward
    float NormalizeReward(float reward) const;
    
    /// @brief Denormalize a reward value
    /// @param normalized_reward Normalized reward
    /// @return Raw reward
    float DenormalizeReward(float normalized_reward) const;

private:
    bool obs_normalization_enabled_;
    bool reward_normalization_enabled_;
    float obs_epsilon_;
    float reward_epsilon_;
    
    std::unique_ptr<RunningMeanStd> obs_running_mean_std_;
    std::unique_ptr<RunningMeanStd> reward_running_mean_std_;
    
    // Buffer for raw observations before normalization
    mutable std::vector<float> raw_obs_buffer_;
};
