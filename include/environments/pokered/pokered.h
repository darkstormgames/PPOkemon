#pragma once

#include <memory>
#include <set>
#include <vector>
#include "torch/envs/env_abstract.h"
#include "emulation/emu_gbc.h"
#include "emulation/inputhandler.h"
#include "emulation/renderer.h"

// Forward declaration for rewards calculator
class PokeredRewards;

class Pokered : public AbstractEnv
{
public:
    Pokered(RenderMode renderMode, unsigned int seed = 0);
    ~Pokered() override;

    virtual int64_t GetObservationSize() const override;
    virtual int64_t GetActionSize() const override;
    virtual std::vector<int64_t> GetObservationShape() const override;

    virtual void ResetImpl() override;
    virtual void RenderImpl() override;
    virtual StepResultRaw StepImpl(const float* action_data, int64_t action_size) override;
    virtual void GetObsData(float* buffer) const override;

    const PokeredRewards::GameState& GetCurrentGameState() const { return current_state_; }
    const PokeredRewards::GameState& GetLastGameState() const { return last_state_; }

private:
    std::unique_ptr<Emulator> emulator;
    std::string biosPath = "./cgb_boot.bin";
    std::string romPath = "./pokered.gbc";
    gambatte::uint_least32_t videoBuffer[160 * 144] = {};
    gambatte::uint_least32_t audioBuffer[37176 + 2064] = {};
    InputHandler inputHandler;
    Renderer renderer;
    std::unique_ptr<PokeredRewards> rewards_calculator_;
    bool render_enabled_ = true;

    // Game state tracking (simplified - managed by rewards calculator)
    PokeredRewards::GameState current_state_;
    PokeredRewards::GameState last_state_;

    // Helper methods  
    void ProcessObservationSIMD(float* buffer) const;
    void InitializeGameState();
    std::string FindFileRecursively(const std::string& baseDir, const std::string& filename) const;
};
