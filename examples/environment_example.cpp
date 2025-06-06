#include <iostream>
#include <memory>
#include <vector>
#include <torch/torch.h>
#include "torch/envs/env_base.h"
#include "torch/envs/env_vectorized.h"
#include "torch/utils/running_mean_std.h"

// Example concrete environment for demonstration
class ExampleEnv : public BaseEnv
{
private:
    std::vector<int64_t> obs_shape_;
    int64_t obs_size_;
    std::normal_distribution<float> noise_dist_;

public:
    ExampleEnv(unsigned int seed = 0) 
        : BaseEnv(seed)
        , obs_shape_({4})  // Simple 4-dimensional observation
        , obs_size_(4)
        , noise_dist_(0.0f, 1.0f)
    {
    }

    std::vector<int64_t> GetObservationShape() const override {
        return obs_shape_;
    }

    int64_t GetObservationSize() const override {
        return obs_size_;
    }

    void GetObsData(float* buffer) const override {
        // Generate random observations for demonstration
        for (int i = 0; i < obs_size_; ++i) {
            buffer[i] = noise_dist_(rng_);
        }
    }

    StepResultRaw Step(int action) override {
        // Simple mock environment step
        episode_steps_++;
        float reward = noise_dist_(rng_);
        episode_reward_ += reward;
        
        bool done = (episode_steps_ >= 100);  // Episode ends after 100 steps
        
        return {reward, done, episode_reward_, episode_steps_};
    }

    void Reset() override {
        AbstractEnv::Reset();
        // Reset any environment-specific state here
    }

    void Render() override {
        std::cout << "Episode " << episode_count_ 
                  << " - Steps: " << episode_steps_ 
                  << " - Reward: " << episode_reward_ << std::endl;
    }
};

int main() {
    std::cout << "PPOkemon Environment Implementation Example" << std::endl;
    std::cout << "==========================================" << std::endl;

    try {
        // 1. Test RunningMeanStd utility
        std::cout << "\n1. Testing RunningMeanStd utility..." << std::endl;
        RunningMeanStd rms({4}, 1e-8f);
        
        // Generate some sample data and update statistics
        auto sample_data = torch::randn({10, 4});
        rms.Update(sample_data);
        
        std::cout << "   Mean: " << rms.GetMean() << std::endl;
        std::cout << "   Variance: " << rms.GetVar() << std::endl;

        // 2. Test single environment with normalization
        std::cout << "\n2. Testing BaseEnv with normalization..." << std::endl;
        ExampleEnv env(42);
        env.SetObservationNormalization(true, 1e-8f);
        env.SetRewardNormalization(true, 1e-8f);
        
        env.Reset();
        
        // Run a few steps
        for (int step = 0; step < 5; ++step) {
            // Get normalized observation
            std::vector<float> obs_buffer(env.GetObservationSize());
            env.GetNormalizedObsData(obs_buffer.data());
            
            // Take a random action
            int action = step % 3;
            auto result = env.Step(action);
            
            // Normalize reward
            float normalized_reward = env.NormalizeReward(result.reward);
            
            std::cout << "   Step " << step 
                      << " - Raw reward: " << result.reward 
                      << " - Normalized: " << normalized_reward << std::endl;
            
            if (result.done) {
                std::cout << "   Episode finished!" << std::endl;
                break;
            }
        }

        // 3. Test vectorized environments
        std::cout << "\n3. Testing VectorizedEnv..." << std::endl;
        std::vector<std::unique_ptr<AbstractEnv>> envs;
        const int num_envs = 4;
        
        for (int i = 0; i < num_envs; ++i) {
            envs.push_back(std::make_unique<ExampleEnv>(i + 100));
        }
        
        VectorizedEnv vec_env(std::move(envs), torch::kCPU);
        
        // Reset all environments
        auto obs_batch = vec_env.Reset();
        std::cout << "   Observation batch shape: " << obs_batch.sizes() << std::endl;
        
        // Take synchronized steps
        std::vector<int> actions(num_envs, 1);
        auto step_results = vec_env.Step(actions);
        
        std::cout << "   Rewards: " << step_results.rewards << std::endl;
        std::cout << "   Done flags: " << step_results.dones << std::endl;

        // 4. Test normalization statistics persistence
        std::cout << "\n4. Testing normalization statistics save/load..." << std::endl;
        
        // Save statistics from the first environment
        env.SaveNormalizationStats("/tmp/obs_stats.bin", "/tmp/reward_stats.bin");
        std::cout << "   Normalization statistics saved" << std::endl;
        
        // Create a new environment and load the statistics
        ExampleEnv env2(999);
        env2.SetObservationNormalization(true, 1e-8f);
        env2.SetRewardNormalization(true, 1e-8f);
        env2.LoadNormalizationStats("/tmp/obs_stats.bin", "/tmp/reward_stats.bin");
        std::cout << "   Normalization statistics loaded into new environment" << std::endl;

        std::cout << "\n✅ All tests completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
