#pragma once

#include <string>
#include <torch/torch.h>

/// @brief Compute the running mean and std using this algorithm:
/// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

class RunningMeanStd
{
public:
    RunningMeanStd(const c10::IntArrayRef shape = {}, const float epsilon = 1e-4f, const torch::Device& device = torch::kCPU);
    ~RunningMeanStd();

    const torch::Tensor& GetMean() const;
    const torch::Tensor& GetVar() const;

    /// @brief Update the mean and var from a batch of size { Nx... }
    /// @param batch The batch of data
    void Update(const torch::Tensor& batch);

    /// @brief Dump this object to a file
    /// @param path Binary file to write the data
    void Save(const std::string& path) const;

    /// @brief Load this object from a file
    /// @param path Binary file to load the data from
    void Load(const std::string& path);

    // These have to be public for device initialization
    torch::Tensor mean;
    torch::Tensor var;

private:
    float count;
    torch::Device device_;
};
