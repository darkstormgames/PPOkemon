#include "environments/pokered/pokered.h"
#include "environments/pokered/pokered_rewards.h"
#include <atomic>
#include <filesystem>
#include <immintrin.h>
#include <stdexcept>
#include <algorithm>

// Memory addresses and state management are now handled by PokeredRewards class

namespace fs = std::filesystem;

Pokered::Pokered(RenderMode renderMode, unsigned int seed)
    : AbstractEnv(seed),
      inputHandler(renderMode),
      renderer(160, 144, 4, renderMode),
      rewards_calculator_(std::make_unique<PokeredRewards>())
{
    // Find ROM and BIOS files recursively
    std::string bootRomPath = FindFileRecursively("..", "cgb_boot.bin");
    romPath = FindFileRecursively("..", "pokered.gbc");

    emulator = std::make_unique<Emulator>(bootRomPath, romPath);

    if (!emulator->isInitialized())
    {
        throw std::runtime_error("Failed to initialize emulator");
    }

    emulator->setInputGetter(&inputHandler);
    if (!renderer.isInitialized())
    {
        throw std::runtime_error("Failed to initialize renderer");
    }
    
    // Initialize game states
    ResetImpl();
}

Pokered::~Pokered()
{
}

int64_t Pokered::GetObservationSize() const
{
    return 160 * 144;
}

int64_t Pokered::GetActionSize() const
{
    return 8;
}

std::vector<int64_t> Pokered::GetObservationShape() const
{
    return {1, 144, 160};
}

void Pokered::GetObsData(float* buffer) const
{
    constexpr int pixels = 160 * 144;
    
    // Check CPU support for AVX2 at runtime
    static const bool has_avx2 = __builtin_cpu_supports("avx2");
    
    if (has_avx2) {
        ProcessObservationSIMD(buffer);
    } else {
        // Fallback scalar implementation
        for (int i = 0; i < pixels; ++i)
        {
            const auto& pixel = videoBuffer[i];
            uint8_t r = (pixel >> 16) & 0xFF;
            uint8_t g = (pixel >> 8) & 0xFF;
            uint8_t b = pixel & 0xFF;
            
            // Convert to grayscale and normalize
            float grayscale = (0.3f * r + 0.59f * g + 0.11f * b) / 255.0f;
            buffer[i] = grayscale;
        }
    }
}

__attribute__((target("avx2,fma")))
void Pokered::ProcessObservationSIMD(float* buffer) const
{
    constexpr int pixels = 160 * 144;
    constexpr int simd_pixels = (pixels / 8) * 8;
    
    const __m256 r_weight = _mm256_set1_ps(0.3f);
    const __m256 g_weight = _mm256_set1_ps(0.59f);
    const __m256 b_weight = _mm256_set1_ps(0.11f);
    const __m256 normalizer = _mm256_set1_ps(1.0f / 255.0f);
    
    int i = 0;
    for (; i < simd_pixels; i += 8)
    {
        // Extract RGB values for 8 pixels at once
        __m256i pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&videoBuffer[i]));
        
        // Extract R, G, B channels
        __m256 r = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(pixels, 16), _mm256_set1_epi32(0xFF)));
        __m256 g = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(pixels, 8), _mm256_set1_epi32(0xFF)));
        __m256 b = _mm256_cvtepi32_ps(_mm256_and_si256(pixels, _mm256_set1_epi32(0xFF)));
        
        // Compute grayscale and normalize
        __m256 grayscale = _mm256_fmadd_ps(r, r_weight, _mm256_fmadd_ps(g, g_weight, _mm256_mul_ps(b, b_weight)));
        __m256 normalized = _mm256_mul_ps(grayscale, normalizer);
        
        _mm256_storeu_ps(&buffer[i], normalized);
    }
    
    // Handle remaining pixels
    for (; i < pixels; ++i)
    {
        const auto& pixel = videoBuffer[i];
        uint8_t r = (pixel >> 16) & 0xFF;
        uint8_t g = (pixel >> 8) & 0xFF;
        uint8_t b = pixel & 0xFF;
        
        float grayscale = (0.3f * r + 0.59f * g + 0.11f * b) / 255.0f;
        buffer[i] = grayscale;
    }
}

void Pokered::ResetImpl()
{
    // Load save state
    std::string romName = std::filesystem::path(romPath).stem().string();
    std::string stateFile = std::filesystem::path(romPath).parent_path().string() + "/" + romName + "_1.gqs";
    emulator->loadState(stateFile);
    
    inputHandler.setInputState(0x00);
    std::size_t samples = 37176;
    emulator->stepFrame(this->videoBuffer, this->audioBuffer, samples);
    renderer.renderFrame(this->videoBuffer);
    
    // Initialize game state
    InitializeGameState();
}

void Pokered::RenderImpl()
{
    if (render_enabled_) {
        renderer.renderFrame(videoBuffer);
    }
}

StepResultRaw Pokered::StepImpl(const float* action_data, int64_t action_size)
{
    // Validate action data
    if (!action_data || action_size != GetActionSize())
    {
        throw std::runtime_error("Invalid action data");
    }

    // Store last state
    last_state_ = current_state_;

    // Convert action data to unsigned integer
    unsigned actionMask = 0;
    for (int i = 0; i < action_size; ++i)
    {
        if (action_data[i] > 0.5f)
        {
            actionMask |= (1u << i);
        }
    }

    inputHandler.setInputState(actionMask);
    std::atomic<bool> test = true;
    inputHandler.poll(test);

    std::size_t samples = 37176;
    // Step the emulator for 3 frames to stay in line with the 24 Frames movement cycles
    emulator->stepFrame(this->videoBuffer, this->audioBuffer, samples);
    emulator->stepFrame(this->videoBuffer, this->audioBuffer, samples);
    emulator->stepFrame(this->videoBuffer, this->audioBuffer, samples);
    
    if (render_enabled_) {
        renderer.renderFrame(this->videoBuffer);
    }
    
    // Update current game state using rewards calculator
    current_state_ = rewards_calculator_->ReadGameState(emulator.get());
    
    // Update movement tracking
    rewards_calculator_->UpdateMovementTracking(current_state_, last_state_, current_episode_length);
    
    // Calculate reward using the rewards calculator with emulator pointer
    float reward = rewards_calculator_->CalculateReward(emulator.get(), last_state_);
    
    // Check for episode termination (could add custom termination conditions here)
    bool done = false; // For now, let the environment run indefinitely
    
    return {reward, done, current_episode_reward + reward, current_episode_length + 1};
}

void Pokered::InitializeGameState()
{
    // Initialize game state using rewards calculator
    rewards_calculator_->InitializeGameState(emulator.get(), current_state_);
    
    // Initialize last state from current state
    last_state_ = current_state_;
}

std::string Pokered::FindFileRecursively(const std::string& baseDir, const std::string& filename) const
{
    std::function<std::string(const fs::path&)> searchRecursive = [&](const fs::path& dir) -> std::string {
        try {
            for (const auto& entry : fs::recursive_directory_iterator(dir)) {
                if (entry.is_regular_file() && entry.path().filename() == filename) {
                    return entry.path().string();
                }
            }
        } catch (const std::exception&) {
            // Directory might not exist or be accessible
        }
        return "";
    };
    
    std::string result = searchRecursive(baseDir);
    if (result.empty()) {
        throw std::runtime_error("Could not find file: " + filename);
    }
    return result;
}