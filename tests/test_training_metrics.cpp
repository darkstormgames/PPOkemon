#include <torch/torch.h>
#include <memory>
#include <vector>
#include "test_utils.h"
#include "../include/torch/training/evaluation/metrics.h"

using namespace PPOkemonTest;
using namespace training;

void test_metrics_basic_functionality() {
    // Test basic episode statistics computation
    std::vector<Metrics::EpisodeStats> episodes;
    
    Metrics::EpisodeStats ep1;
    ep1.total_reward = 100.5f;
    ep1.total_steps = 250;
    ep1.episode_length = 250.0f;
    ep1.success = true;
    episodes.push_back(ep1);
    
    Metrics::EpisodeStats ep2;
    ep2.total_reward = 150.0f;
    ep2.total_steps = 300;
    ep2.episode_length = 300.0f;
    ep2.success = true;
    episodes.push_back(ep2);
    
    auto stats = Metrics::ComputeEpisodeStats(episodes);
    ASSERT_NEAR(stats.mean_reward, 125.25f, 0.01f);
    ASSERT_NEAR(stats.mean_episode_length, 275.0f, 0.01f);
    ASSERT_EQ(stats.num_episodes, 2);
    ASSERT_NEAR(stats.success_rate, 1.0f, 0.01f);
}

void test_metrics_reward_tracking() {
    // Test reward computation functions
    std::vector<float> rewards = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    
    float mean_reward = Metrics::ComputeMeanReward(rewards);
    ASSERT_NEAR(mean_reward, 3.0f, 0.01f);
    
    float std_reward = Metrics::ComputeStdReward(rewards);
    ASSERT_TRUE(std_reward > 0.0f);
    
    auto reward_range = Metrics::ComputeRewardRange(rewards);
    ASSERT_NEAR(reward_range.first, 1.0f, 0.01f);  // min
    ASSERT_NEAR(reward_range.second, 5.0f, 0.01f); // max
}

void test_metrics_custom_metrics() {
    // Test custom metric registration
    auto exploration_metric = [](const std::vector<Metrics::EpisodeStats>& episodes) -> float {
        (void)episodes; // Suppress unused parameter warning
        return 0.1f; // Mock exploration rate
    };
    
    ASSERT_NO_THROW(Metrics::RegisterCustomMetric("exploration_rate", exploration_metric));
    
    // Create some episodes to test custom metrics
    std::vector<Metrics::EpisodeStats> episodes;
    Metrics::EpisodeStats ep;
    ep.total_reward = 100.0f;
    ep.episode_length = 200.0f;
    episodes.push_back(ep);
    
    auto custom_stats = Metrics::ComputeCustomMetrics(episodes);
    ASSERT_TRUE(custom_stats.find("exploration_rate") != custom_stats.end());
    ASSERT_NEAR(custom_stats["exploration_rate"], 0.1f, 0.01f);
}

void test_metrics_large_dataset() {
    // Test with large amount of data using static methods
    const int num_episodes = 1000;
    std::vector<Metrics::EpisodeStats> episodes;
    
    for (int i = 0; i < num_episodes; ++i) {
        Metrics::EpisodeStats episode;
        episode.total_reward = 100.0f + (i % 100);
        episode.total_steps = 200 + (i % 50);
        episode.success = (i % 10) == 0;
        episodes.push_back(episode);
    }
    
    auto stats = Metrics::ComputeEpisodeStats(episodes);
    ASSERT_EQ(stats.num_episodes, num_episodes);
    ASSERT_TRUE(stats.mean_reward > 0);
    ASSERT_TRUE(stats.mean_episode_length > 0);
}

int main() {
    TestSuite suite("PPOkemon Training Metrics Tests");
    
    suite.AddTest("Basic Functionality", test_metrics_basic_functionality);
    suite.AddTest("Reward Tracking", test_metrics_reward_tracking);
    suite.AddTest("Custom Metrics", test_metrics_custom_metrics);
    suite.AddTest("Large Dataset", test_metrics_large_dataset);
    
    bool all_passed = suite.RunAll();
    
    int passed, failed, total;
    suite.GetStats(passed, failed, total);
    
    std::cout << "\nFinal Results: " << passed << "/" << total << " tests passed" << std::endl;
    
    return all_passed ? 0 : 1;
}
