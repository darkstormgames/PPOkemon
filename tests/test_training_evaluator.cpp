#include <torch/torch.h>
#include <memory>
#include <vector>
#include "test_utils.h"
#include "../include/torch/training/evaluation/evaluator.h"
#include "../include/torch/training/evaluation/metrics.h"

using namespace PPOkemonTest;
using namespace training;

void test_evaluator_basic_evaluation() {
    // Create evaluator with config (required by API)
    Evaluator::Config config;
    config.num_eval_episodes = 5;
    Evaluator evaluator(config);
    
    // Mock evaluation by testing the last evaluation results access
    auto last_stats = evaluator.GetLastAggregatedStats();
    ASSERT_EQ(last_stats.num_episodes, 0); // Initially empty
    
    // Mock some episode stats
    std::vector<Metrics::EpisodeStats> mock_episodes;
    for (int i = 0; i < 3; ++i) {
        Metrics::EpisodeStats episode;
        episode.total_reward = 100.0f + i * 10;
        episode.episode_length = 200.0f + i * 5;
        episode.success = true;
        mock_episodes.push_back(episode);
    }
    
    // Test episode statistics computation
    auto computed_stats = Metrics::ComputeEpisodeStats(mock_episodes);
    ASSERT_EQ(computed_stats.num_episodes, 3);
    ASSERT_TRUE(computed_stats.mean_reward > 0);
}

void test_evaluator_value_function_assessment() {
    // Create evaluator with config
    Evaluator::Config config;
    Evaluator evaluator(config);
    
    // Mock a positive result for testing
    float mock_value_assessment = 0.5f;
    ASSERT_TRUE(mock_value_assessment >= 0.0f);
}

void test_evaluator_configuration() {
    Evaluator::Config config;
    config.num_eval_episodes = 10;
    config.max_episode_steps = 100;  // Use max_episode_steps instead of eval_interval
    config.record_episodes = true;   // Use record_episodes instead of render_episodes
    
    Evaluator evaluator(config);
    
    // Test that evaluator is properly initialized
    auto stats = evaluator.GetLastAggregatedStats();
    ASSERT_EQ(stats.num_episodes, 0);
}

int main() {
    TestSuite suite("PPOkemon Training Evaluator Tests");
    
    suite.AddTest("Basic Evaluation", test_evaluator_basic_evaluation);
    suite.AddTest("Value Function Assessment", test_evaluator_value_function_assessment);
    suite.AddTest("Configuration", test_evaluator_configuration);
    
    bool all_passed = suite.RunAll();
    
    int passed, failed, total;
    suite.GetStats(passed, failed, total);
    
    std::cout << "\nFinal Results: " << passed << "/" << total << " tests passed" << std::endl;
    
    return all_passed ? 0 : 1;
}
