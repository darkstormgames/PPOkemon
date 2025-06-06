#include "torch/utils/running_mean_std.h"

RunningMeanStd::RunningMeanStd(const c10::IntArrayRef shape, const float epsilon, const torch::Device& device)
    : device_(device)
{
    mean = torch::zeros(shape, torch::TensorOptions().device(device_)).set_requires_grad(false);
    var = torch::ones(shape, torch::TensorOptions().device(device_)).set_requires_grad(false);
    count = epsilon;
}

RunningMeanStd::~RunningMeanStd()
{

}

const torch::Tensor& RunningMeanStd::GetMean() const
{
    return mean;
}

const torch::Tensor& RunningMeanStd::GetVar() const
{
    return var;
}

void RunningMeanStd::Update(const torch::Tensor& batch)
{
    torch::NoGradGuard no_grad;
    torch::Tensor batch_on_device = (batch.device() == device_) ? batch : batch.to(device_);

    torch::Tensor delta = batch_on_device.mean(0) - mean;
    const int64_t N = batch_on_device.size(0);
    const float tot_count = count + N;

    mean = mean + delta * N / tot_count;
    var = (var * count + batch_on_device.var(0, false) * N + delta * delta * count * N / tot_count) / tot_count;
    count = tot_count;
}

void RunningMeanStd::Save(const std::string& path) const
{
    // Ensure tensors are on CPU before saving
    torch::save({ mean.cpu(), var.cpu(), torch::zeros({}).cpu() + count }, path);
}

void RunningMeanStd::Load(const std::string& path)
{
    std::vector<torch::Tensor> data;
    torch::load(data, path);
    mean = data[0].to(device_);
    var = data[1].to(device_);
    count = data[2].item<float>();
}
