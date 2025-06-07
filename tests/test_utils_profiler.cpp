#include <iostream>
#include <chrono>
#include <thread>
#include <filesystem>
#include "torch/utils/profiler.h"
#include "test_utils.h"

using namespace PPOkemonTest;

void test_profiler_manual_timing() {
    auto& profiler = utils::Profiler::Instance();
    
    // Test manual timing
    profiler.StartTimer("manual_test");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    profiler.StopTimer("manual_test");
    
    double avg_time = profiler.GetAverageTime("manual_test");
    ASSERT_TRUE(avg_time > 0.005); // Should be at least 5ms
    ASSERT_EQ(profiler.GetCallCount("manual_test"), 1);
}

void test_profiler_scoped_timing() {
    auto& profiler = utils::Profiler::Instance();
    
    // Test scoped timing
    {
        PROFILE_SCOPE("scoped_test");
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    
    ASSERT_TRUE(profiler.GetAverageTime("scoped_test") > 0.003);
    ASSERT_EQ(profiler.GetCallCount("scoped_test"), 1);
}

void test_profiler_function_timing() {
    auto& profiler = utils::Profiler::Instance();
    
    // Clear any existing data first
    profiler.Clear();
    
    auto do_work = []() {
        PROFILE_SCOPE("lambda_work");  // Use explicit name instead of __FUNCTION__
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    };
    
    // Run function multiple times
    for (int i = 0; i < 3; ++i) {
        do_work();
    }
    
    ASSERT_EQ(profiler.GetCallCount("lambda_work"), 3);
    ASSERT_TRUE(profiler.GetTotalTime("lambda_work") > 0.010);
}

void test_profiler_direct_recording() {
    auto& profiler = utils::Profiler::Instance();
    
    // Test direct time recording
    profiler.RecordTime("direct_record", 0.020); // 20ms
    
    ASSERT_NEAR(profiler.GetAverageTime("direct_record"), 0.020, 0.001);
    ASSERT_EQ(profiler.GetCallCount("direct_record"), 1);
}

void test_profiler_save_report() {
    auto& profiler = utils::Profiler::Instance();
    
    // Add some test data
    profiler.RecordTime("test_operation", 0.015);
    
    std::string report_path = "./profiler_report.txt";
    ASSERT_NO_THROW(profiler.SaveReport(report_path));
    ASSERT_TRUE(std::filesystem::exists(report_path));
    
    // Cleanup
    std::filesystem::remove(report_path);
}

int main() {
    TestSuite suite("PPOkemon Profiler Tests");
    
    suite.AddTest("Manual Timing", test_profiler_manual_timing);
    suite.AddTest("Scoped Timing", test_profiler_scoped_timing);
    suite.AddTest("Function Timing", test_profiler_function_timing);
    suite.AddTest("Direct Recording", test_profiler_direct_recording);
    suite.AddTest("Save Report", test_profiler_save_report);
    
    bool all_passed = suite.RunAll();
    
    int passed, failed, total;
    suite.GetStats(passed, failed, total);
    
    std::cout << "\nFinal Results: " << passed << "/" << total << " tests passed" << std::endl;
    
    return all_passed ? 0 : 1;
}