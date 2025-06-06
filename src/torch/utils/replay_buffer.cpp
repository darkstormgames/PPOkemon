#include "torch/utils/replay_buffer.h"
#include <torch/torch.h>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace utils {

ReplayBuffer::ReplayBuffer(size_t capacity, unsigned int seed)
    : capacity_(capacity), current_size_(0), position_(0), use_priorities_(false),
      rng_(seed), uniform_dist_(0.0f, 1.0f)
{
    // Reserve space for better performance
    states_.reserve(capacity);
    actions_.reserve(capacity);
    rewards_.reserve(capacity);
    next_states_.reserve(capacity);
    dones_.reserve(capacity);
    priorities_.reserve(capacity);
}

void ReplayBuffer::Push(const torch::Tensor& state, const torch::Tensor& action, 
                       float reward, const torch::Tensor& next_state, bool done) {
    if (current_size_ < capacity_) {
        // Buffer not full yet - append
        states_.push_back(state.detach().clone());
        actions_.push_back(action.detach().clone());
        rewards_.push_back(reward);
        next_states_.push_back(next_state.detach().clone());
        dones_.push_back(done);
        priorities_.push_back(1.0f); // Default priority
        current_size_++;
    } else {
        // Buffer full - circular replacement
        states_[position_] = state.detach().clone();
        actions_[position_] = action.detach().clone();
        rewards_[position_] = reward;
        next_states_[position_] = next_state.detach().clone();
        dones_[position_] = done;
        priorities_[position_] = 1.0f;
    }
    
    position_ = (position_ + 1) % capacity_;
}

void ReplayBuffer::PushBatch(const std::vector<Experience>& experiences) {
    for (const auto& exp : experiences) {
        Push(exp.state, exp.action, exp.reward, exp.next_state, exp.done);
    }
}

Transition ReplayBuffer::Sample(size_t batch_size) {
    if (batch_size > current_size_) {
        throw std::invalid_argument("Batch size larger than buffer size");
    }
    
    auto indices = SampleIndices(batch_size);
    
    Transition batch;
    batch.states.reserve(batch_size);
    batch.actions.reserve(batch_size);
    batch.rewards.reserve(batch_size);
    batch.next_states.reserve(batch_size);
    batch.dones.reserve(batch_size);
    
    for (size_t idx : indices) {
        batch.states.push_back(states_[idx]);
        batch.actions.push_back(actions_[idx]);
        batch.rewards.push_back(rewards_[idx]);
        batch.next_states.push_back(next_states_[idx]);
        batch.dones.push_back(dones_[idx]);
    }
    
    return batch;
}

Transition ReplayBuffer::SampleSequential(size_t batch_size, size_t sequence_length) {
    if (batch_size * sequence_length > current_size_) {
        throw std::invalid_argument("Requested sequence batch size too large");
    }
    
    Transition batch;
    batch.states.reserve(batch_size * sequence_length);
    batch.actions.reserve(batch_size * sequence_length);
    batch.rewards.reserve(batch_size * sequence_length);
    batch.next_states.reserve(batch_size * sequence_length);
    batch.dones.reserve(batch_size * sequence_length);
    
    std::uniform_int_distribution<size_t> dist(0, current_size_ - sequence_length);
    
    for (size_t b = 0; b < batch_size; ++b) {
        size_t start_idx = dist(rng_);
        
        for (size_t s = 0; s < sequence_length; ++s) {
            size_t idx = start_idx + s;
            batch.states.push_back(states_[idx]);
            batch.actions.push_back(actions_[idx]);
            batch.rewards.push_back(rewards_[idx]);
            batch.next_states.push_back(next_states_[idx]);
            batch.dones.push_back(dones_[idx]);
        }
    }
    
    return batch;
}

Transition ReplayBuffer::SamplePrioritized(size_t batch_size, float alpha, float beta) {
    if (!use_priorities_) {
        // Initialize priorities if not already done
        use_priorities_ = true;
        std::fill(priorities_.begin(), priorities_.end(), 1.0f);
    }
    
    auto indices = SamplePrioritizedIndices(batch_size, alpha, beta);
    
    Transition batch;
    batch.states.reserve(batch_size);
    batch.actions.reserve(batch_size);
    batch.rewards.reserve(batch_size);
    batch.next_states.reserve(batch_size);
    batch.dones.reserve(batch_size);
    
    for (size_t idx : indices) {
        batch.states.push_back(states_[idx]);
        batch.actions.push_back(actions_[idx]);
        batch.rewards.push_back(rewards_[idx]);
        batch.next_states.push_back(next_states_[idx]);
        batch.dones.push_back(dones_[idx]);
    }
    
    return batch;
}

void ReplayBuffer::UpdatePriorities(const std::vector<size_t>& indices, 
                                   const std::vector<float>& priorities) {
    if (indices.size() != priorities.size()) {
        throw std::invalid_argument("Indices and priorities vectors must have same size");
    }
    
    use_priorities_ = true;
    
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < current_size_) {
            priorities_[indices[i]] = std::max(priorities[i], 1e-6f); // Avoid zero priorities
        }
    }
}

void ReplayBuffer::Clear() {
    states_.clear();
    actions_.clear();
    rewards_.clear();
    next_states_.clear();
    dones_.clear();
    priorities_.clear();
    
    current_size_ = 0;
    position_ = 0;
    use_priorities_ = false;
}

void ReplayBuffer::Save(const std::string& path) const {
    // Save buffer data
    std::vector<torch::Tensor> tensors;
    
    // Pack all states, actions, and next_states
    for (size_t i = 0; i < current_size_; ++i) {
        tensors.push_back(states_[i]);
    }
    for (size_t i = 0; i < current_size_; ++i) {
        tensors.push_back(actions_[i]);
    }
    for (size_t i = 0; i < current_size_; ++i) {
        tensors.push_back(next_states_[i]);
    }
    
    torch::save(tensors, path + ".tensors");
    
    // Save metadata
    std::ofstream meta_file(path + ".meta");
    meta_file << capacity_ << "\n";
    meta_file << current_size_ << "\n";
    meta_file << position_ << "\n";
    meta_file << use_priorities_ << "\n";
    
    // Save rewards, dones, and priorities
    for (size_t i = 0; i < current_size_; ++i) {
        meta_file << rewards_[i] << " ";
    }
    meta_file << "\n";
    
    for (size_t i = 0; i < current_size_; ++i) {
        meta_file << dones_[i] << " ";
    }
    meta_file << "\n";
    
    for (size_t i = 0; i < current_size_; ++i) {
        meta_file << priorities_[i] << " ";
    }
    meta_file << "\n";
    
    meta_file.close();
}

void ReplayBuffer::Load(const std::string& path) {
    // Load tensors
    std::vector<torch::Tensor> tensors;
    torch::load(tensors, path + ".tensors");
    
    // Load metadata
    std::ifstream meta_file(path + ".meta");
    if (!meta_file) {
        throw std::runtime_error("Failed to open replay buffer metadata file");
    }
    
    meta_file >> capacity_ >> current_size_ >> position_ >> use_priorities_;
    
    // Resize vectors
    states_.resize(current_size_);
    actions_.resize(current_size_);
    next_states_.resize(current_size_);
    rewards_.resize(current_size_);
    dones_.resize(current_size_);
    priorities_.resize(current_size_);
    
    // Load tensors
    for (size_t i = 0; i < current_size_; ++i) {
        states_[i] = tensors[i];
    }
    for (size_t i = 0; i < current_size_; ++i) {
        actions_[i] = tensors[current_size_ + i];
    }
    for (size_t i = 0; i < current_size_; ++i) {
        next_states_[i] = tensors[2 * current_size_ + i];
    }
    
    // Load rewards
    for (size_t i = 0; i < current_size_; ++i) {
        meta_file >> rewards_[i];
    }
    
    // Load dones
    for (size_t i = 0; i < current_size_; ++i) {
        int done;
        meta_file >> done;
        dones_[i] = static_cast<bool>(done);
    }
    
    // Load priorities
    for (size_t i = 0; i < current_size_; ++i) {
        meta_file >> priorities_[i];
    }
    
    meta_file.close();
}

std::vector<size_t> ReplayBuffer::SampleIndices(size_t batch_size) {
    std::vector<size_t> indices;
    indices.reserve(batch_size);
    
    std::uniform_int_distribution<size_t> dist(0, current_size_ - 1);
    for (size_t i = 0; i < batch_size; ++i) {
        indices.push_back(dist(rng_));
    }
    
    return indices;
}

std::vector<size_t> ReplayBuffer::SamplePrioritizedIndices(size_t batch_size, float alpha, float beta) {
    // Calculate probability distribution based on priorities
    std::vector<float> probs(current_size_);
    float total_priority = 0.0f;
    
    for (size_t i = 0; i < current_size_; ++i) {
        probs[i] = std::pow(priorities_[i], alpha);
        total_priority += probs[i];
    }
    
    // Normalize probabilities
    for (size_t i = 0; i < current_size_; ++i) {
        probs[i] /= total_priority;
    }
    
    // Sample indices based on probabilities
    std::vector<size_t> indices;
    indices.reserve(batch_size);
    
    for (size_t b = 0; b < batch_size; ++b) {
        float rand_val = uniform_dist_(rng_);
        float cumulative_prob = 0.0f;
        
        for (size_t i = 0; i < current_size_; ++i) {
            cumulative_prob += probs[i];
            if (rand_val <= cumulative_prob) {
                indices.push_back(i);
                break;
            }
        }
    }
    
    return indices;
}

float ReplayBuffer::ComputeMaxWeight(float beta) const {
    if (!use_priorities_) return 1.0f;
    
    float min_prob = *std::min_element(priorities_.begin(), priorities_.begin() + current_size_);
    return std::pow(current_size_ * min_prob, -beta);
}

} // namespace utils
