#pragma once

#include <iostream>
#include <string>
#include <functional>
#include <vector>
#include <chrono>
#include <exception>

// Simple test framework for PPOkemon
namespace PPOkemonTest {

// Test result structure
struct TestResult {
    bool passed;
    std::string message;
    double duration_ms;
};

// Test case structure
struct TestCase {
    std::string name;
    std::function<void()> test_func;
    bool critical = false;  // If true, failure stops all tests
};

// Test suite class
class TestSuite {
private:
    std::string suite_name_;
    std::vector<TestCase> test_cases_;
    int passed_count_ = 0;
    int failed_count_ = 0;
    int total_count_ = 0;

public:
    explicit TestSuite(const std::string& name) : suite_name_(name) {}

    // Add a test case
    void AddTest(const std::string& name, std::function<void()> test_func, bool critical = false) {
        test_cases_.push_back({name, test_func, critical});
    }

    // Run all tests in the suite
    bool RunAll() {
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "Running Test Suite: " << suite_name_ << std::endl;
        std::cout << std::string(50, '=') << std::endl;

        bool all_passed = true;
        total_count_ = test_cases_.size();

        for (const auto& test_case : test_cases_) {
            auto result = RunSingleTest(test_case);
            
            if (result.passed) {
                passed_count_++;
                std::cout << "✅ " << test_case.name << " (" << result.duration_ms << "ms)" << std::endl;
            } else {
                failed_count_++;
                all_passed = false;
                std::cout << "❌ " << test_case.name << " - " << result.message 
                         << " (" << result.duration_ms << "ms)" << std::endl;
                
                if (test_case.critical) {
                    std::cout << "🛑 Critical test failed, stopping suite execution." << std::endl;
                    break;
                }
            }
        }

        PrintSummary();
        return all_passed;
    }

    // Get test statistics
    void GetStats(int& passed, int& failed, int& total) const {
        passed = passed_count_;
        failed = failed_count_;
        total = total_count_;
    }

private:
    TestResult RunSingleTest(const TestCase& test_case) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            test_case.test_func();
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            return {true, "", duration.count() / 1000.0};
        } catch (const std::exception& e) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            return {false, e.what(), duration.count() / 1000.0};
        } catch (...) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            return {false, "Unknown exception", duration.count() / 1000.0};
        }
    }

    void PrintSummary() {
        std::cout << "\n" << std::string(50, '-') << std::endl;
        std::cout << "Test Suite Summary: " << suite_name_ << std::endl;
        std::cout << "Total: " << total_count_ << " | Passed: " << passed_count_ 
                  << " | Failed: " << failed_count_ << std::endl;
        
        if (failed_count_ == 0) {
            std::cout << "🎉 All tests passed!" << std::endl;
        } else {
            std::cout << "⚠️  " << failed_count_ << " test(s) failed." << std::endl;
        }
        std::cout << std::string(50, '-') << std::endl;
    }
};

// Assertion macros for easier testing
#define ASSERT_TRUE(condition) \
    do { \
        if (!(condition)) { \
            throw std::runtime_error("Assertion failed: " #condition); \
        } \
    } while(0)

#define ASSERT_FALSE(condition) \
    do { \
        if (condition) { \
            throw std::runtime_error("Assertion failed: " #condition " should be false"); \
        } \
    } while(0)

#define ASSERT_EQ(expected, actual) \
    do { \
        if ((expected) != (actual)) { \
            throw std::runtime_error("Assertion failed: expected " + std::to_string(expected) + \
                                   " but got " + std::to_string(actual)); \
        } \
    } while(0)

#define ASSERT_NEAR(expected, actual, tolerance) \
    do { \
        auto diff = std::abs((expected) - (actual)); \
        if (diff > (tolerance)) { \
            throw std::runtime_error("Assertion failed: |" + std::to_string(expected) + \
                                   " - " + std::to_string(actual) + "| = " + std::to_string(diff) + \
                                   " > " + std::to_string(tolerance)); \
        } \
    } while(0)

#define ASSERT_THROW(expression, exception_type) \
    do { \
        bool threw_correct = false; \
        try { \
            expression; \
        } catch (const exception_type&) { \
            threw_correct = true; \
        } catch (...) { \
            throw std::runtime_error("Assertion failed: " #expression " threw wrong exception type"); \
        } \
        if (!threw_correct) { \
            throw std::runtime_error("Assertion failed: " #expression " did not throw " #exception_type); \
        } \
    } while(0)

#define ASSERT_NO_THROW(expression) \
    do { \
        try { \
            expression; \
        } catch (...) { \
            throw std::runtime_error("Assertion failed: " #expression " threw an exception"); \
        } \
    } while(0)

} // namespace PPOkemonTest
