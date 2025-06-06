Based on your codebase structure and the discussion about modularizing the arguments system, here's a suggested file and folder restructure for your `torch_rl` modules:

## Proposed Structure:

```
torch_rl/
├── common/
│   ├── args/
│   │   ├── base_args.h
│   │   ├── algorithm_args.h
│   │   ├── network_args.h
│   │   ├── training_args.h
│   │   └── arg_parser.h/.cpp
│   ├── utils/
│   │   ├── running_mean_std.h/.cpp
│   │   ├── tensor_utils.h/.cpp
│   │   └── logger.h/.cpp
│   └── types.h  # Common type definitions, enums (RenderMode, etc.)
│
├── networks/
│   ├── base/
│   │   ├── network_base.h
│   │   └── initializers.h/.cpp
│   ├── mlp/
│   │   ├── mlp.h
│   │   └── mlp.cpp
│   ├── cnn/
│   │   ├── cnn_body.h
│   │   └── cnn_body.cpp
│   └── policy_networks/
│       ├── actor_critic.h
│       └── actor_critic.cpp
│
├── algorithms/
│   ├── base/
│   │   ├── algorithm_base.h
│   │   └── rollout_buffer.h/.cpp
│   ├── ppo/
│   │   ├── ppo.h
│   │   ├── ppo.cpp
│   │   └── ppo_args.h
│   ├── appo/
│   │   ├── appo.h
│   │   ├── appo.cpp
│   │   └── appo_args.h
│   └── dqn/  # Future expansion
│       ├── dqn.h
│       └── dqn.cpp
│
├── envs/
│   ├── base/
│   │   ├── abstract_env.h
│   │   ├── vectorized_env.h/.cpp
│   │   └── env_args.h
│   ├── wrappers/
│   │   ├── normalization_wrapper.h/.cpp
│   │   ├── frame_stack_wrapper.h/.cpp
│   │   └── reward_wrapper.h/.cpp
│   └── implementations/  # Actual environments go here
│       └── (empty - environments stay in main src/emulation/)
│
└── training/
    ├── trainer.h/.cpp  # High-level training orchestration
    ├── callbacks/
    │   ├── callback_base.h
    │   ├── checkpoint_callback.h/.cpp
    │   └── logging_callback.h/.cpp
    └── evaluation/
        ├── evaluator.h/.cpp
        └── metrics.h/.cpp
```

## Key Benefits:

### 1. **Clear Separation of Concerns**
- **networks/**: All neural network architectures
- **algorithms/**: RL algorithms (PPO, APPO, future DQN, etc.)
- **envs/**: Environment interfaces and wrappers
- **common/**: Shared utilities and base classes
- **training/**: Training orchestration and evaluation

### 2. **Modular Network Architecture**
Split your current mlp.cpp into:

```cpp
// networks/base/network_base.h
class NetworkBase : public torch::nn::Module {
public:
    virtual void InitOrtho(float gain) = 0;
};

// networks/mlp/mlp.h
class MLPImpl : public NetworkBase {
    // Current MLP implementation
};

// networks/cnn/cnn_body.h
class CNNBodyImpl : public NetworkBase {
    // Current CNN implementation
};

// networks/policy_networks/actor_critic.h
class ActorCritic : public torch::nn::Module {
    // Combines feature extractor + policy/value heads
private:
    std::shared_ptr<NetworkBase> feature_extractor;
    torch::nn::Linear policy_head;
    torch::nn::Linear value_head;
};
```

### 3. **Algorithm Organization**
Each algorithm gets its own directory:

```cpp
// algorithms/ppo/ppo_args.h
struct PPOSpecificArgs {
    float clip_range = 0.15f;
    uint64_t n_steps = 2048;
    uint64_t n_epochs = 8;
    // PPO-specific parameters only
};

// algorithms/base/algorithm_base.h
class AlgorithmBase {
public:
    virtual std::vector<std::pair<uint64_t, float>> Learn(...) = 0;
    virtual std::vector<std::pair<uint64_t, float>> Play(...) = 0;
};
```

### 4. **Flexible Args System**
```cpp
// common/args/base_args.h
struct BaseArgs {
    unsigned int seed;
    std::string exp_path;
    bool use_gpu_if_available;
};

// common/args/network_args.h
struct NetworkArgs {
    enum class NetworkType { MLP, CNN, CUSTOM };
    NetworkType type = NetworkType::CNN;
    
    // CNN specific
    int64_t cnn_out_features = 256;
    
    // MLP specific
    int64_t mlp_hidden_units = 128;
    
    // Common
    bool ortho_init = true;
    float init_gain = sqrt(2.0f);
};

// common/args/algorithm_args.h
struct AlgorithmArgs : public BaseArgs {
    float gamma = 0.995f;
    float lr = 3e-4f;
    uint64_t batch_size = 256;
    float max_grad_norm = 0.5f;
};
```

### 5. **Environment Wrappers**
```cpp
// envs/wrappers/frame_stack_wrapper.h
template<typename EnvType>
class FrameStackWrapper : public AbstractEnv {
    // Wraps environment to stack frames
};

// envs/wrappers/normalization_wrapper.h
template<typename EnvType>
class NormalizationWrapper : public AbstractEnv {
    // Handles observation/reward normalization
};
```

## Migration Strategy:

1. **Phase 1**: Create new directory structure, move files one module at a time
2. **Phase 2**: Refactor imports and CMakeLists.txt
3. **Phase 3**: Split large files (like mlp.cpp) into appropriate modules
4. **Phase 4**: Implement new args system gradually

This structure will make your codebase more maintainable, easier to extend with new algorithms/networks, and clearer for other developers to understand.








You're absolutely right! Having separate folders for single files would be overkill. Here's a more practical structure that keeps related `.h` and `.cpp` files together:

## Updated Proposed Structure:

```
torch_rl/
├── common/
│   ├── args.h
│   ├── args.cpp
│   ├── running_mean_std.h
│   ├── running_mean_std.cpp
│   ├── tensor_utils.h
│   ├── tensor_utils.cpp
│   └── types.h  # Common type definitions, enums (RenderMode, etc.)
│
├── networks/
│   ├── network_base.h
│   ├── network_base.cpp
│   ├── mlp.h
│   ├── mlp.cpp
│   ├── cnn_body.h
│   ├── cnn_body.cpp
│   ├── actor_critic.h
│   └── actor_critic.cpp
│
├── algorithms/
│   ├── algorithm_base.h
│   ├── algorithm_base.cpp
│   ├── rollout_buffer.h
│   ├── rollout_buffer.cpp
│   ├── ppo.h
│   ├── ppo.cpp
│   ├── ppo_args.h
│   ├── appo.h
│   ├── appo.cpp
│   └── appo_args.h
│
├── envs/
│   ├── abstract_env.h
│   ├── vectorized_env.h
│   ├── vectorized_env.cpp
│   └── env_args.h
│
└── utils/
    ├── logger.h
    ├── logger.cpp
    ├── checkpoint_manager.h
    └── checkpoint_manager.cpp
```

## Key Changes from Previous Suggestion:

### 1. **Flattened Structure**
- No more subdirectories for single files
- Related `.h` and `.cpp` files sit side by side
- Easier navigation and clearer relationships

### 2. **Simplified Organization**
```cpp
// networks/mlp.h
#pragma once
#include "network_base.h"

class MLPImpl : public NetworkBase {
    // MLP interface
};
using MLP = torch::nn::ModuleHolder<MLPImpl>;

// networks/mlp.cpp
#include "mlp.h"
// MLP implementation
```

### 3. **Modular Args System Without Deep Nesting**
```cpp
// common/args.h
#pragma once

struct BaseArgs {
    unsigned int seed;
    std::string exp_path;
    bool use_gpu_if_available;
};

struct NetworkArgs {
    enum class NetworkType { MLP, CNN };
    NetworkType type = NetworkType::CNN;
    int64_t cnn_out_features = 256;
    int64_t mlp_hidden_units = 128;
    bool ortho_init = true;
};

struct TrainingArgs {
    uint64_t total_timesteps_per_iteration = 1 << 20;
    int max_iterations = 0;
    uint64_t log_interval = 5;
};

// algorithms/ppo_args.h
#pragma once
#include "../common/args.h"

struct PPOArgs : public BaseArgs {
    // Algorithm base parameters
    float gamma = 0.995f;
    float lr = 3e-4f;
    uint64_t batch_size = 256;
    
    // PPO specific
    uint64_t n_steps = 2048;
    uint64_t n_epochs = 8;
    float clip_range = 0.15f;
    float vf_coef = 0.5f;
    float ent_coef = 0.01f;
    
    // Network configuration
    NetworkArgs network;
    
    // Training configuration
    TrainingArgs training;
};
```

### 4. **Practical File Splits**

For your current `mlp.cpp`, split it into:

```cpp
// networks/network_base.h
#pragma once
#include <torch/torch.h>

class NetworkBase : public torch::nn::Module {
public:
    virtual ~NetworkBase() = default;
    virtual void InitOrtho(float gain) = 0;
    virtual int64_t GetOutputSize() const = 0;
};

// networks/mlp.h
#pragma once
#include "network_base.h"

class MLPImpl : public NetworkBase {
public:
    MLPImpl(int64_t num_in, int64_t num_hidden, int64_t out_num);
    ~MLPImpl();
    
    torch::Tensor forward(const torch::Tensor& in);
    void InitOrtho(float gain) override;
    void InitOrtho(float gain_backbone, float gain_out);
    int64_t GetOutputSize() const override { return out_num_; }
    
private:
    torch::nn::Linear l1{nullptr}, l2{nullptr}, out_layer{nullptr};
    int64_t out_num_;
};
TORCH_MODULE(MLP);

// networks/cnn_body.h
#pragma once
#include "network_base.h"

class CNNBodyImpl : public NetworkBase {
public:
    CNNBodyImpl(int64_t num_input_channels, int64_t input_height, 
                int64_t input_width, int conv_out_size);
    ~CNNBodyImpl();
    
    torch::Tensor forward(const torch::Tensor& in);
    void InitOrtho(float gain) override;
    int64_t GetOutputSize() const override { return output_size; }
    
private:
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
    torch::nn::Linear linear{nullptr};
    int64_t input_height_, input_width_, output_size;
};
TORCH_MODULE(CNNBody);

// networks/actor_critic.h
#pragma once
#include "network_base.h"

class ActorCritic : public torch::nn::Module {
public:
    ActorCritic(std::shared_ptr<NetworkBase> feature_extractor,
                int64_t action_size);
    
    std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& obs);
    torch::Tensor PredictValues(const torch::Tensor& obs);
    // ... other methods
    
private:
    std::shared_ptr<NetworkBase> feature_extractor_;
    torch::nn::Linear policy_head_{nullptr};
    torch::nn::Linear value_head_{nullptr};
};
```

### 5. **Simplified CMakeLists.txt**

```cmake
# In your main CMakeLists.txt or a torch_rl/CMakeLists.txt
file(GLOB TORCH_RL_SOURCES
  ${CMAKE_SOURCE_DIR}/src/torch_rl/common/*.cpp
  ${CMAKE_SOURCE_DIR}/src/torch_rl/networks/*.cpp
  ${CMAKE_SOURCE_DIR}/src/torch_rl/algorithms/*.cpp
  ${CMAKE_SOURCE_DIR}/src/torch_rl/envs/*.cpp
  ${CMAKE_SOURCE_DIR}/src/torch_rl/utils/*.cpp
)

# Include directories
target_include_directories(drl_test PRIVATE
  ${CMAKE_SOURCE_DIR}/src/torch_rl
)
```

## Benefits of This Structure:

1. **Clear and Simple**: No deep nesting, easy to find files
2. **Logical Grouping**: Related functionality stays together
3. **Easy to Extend**: Adding new algorithms/networks is straightforward
4. **Practical**: Reflects how the code is actually used
5. **Better IDE Support**: Most IDEs handle flat structures better than deep nesting

This structure maintains modularity while being practical for your codebase size and avoiding unnecessary complexity.