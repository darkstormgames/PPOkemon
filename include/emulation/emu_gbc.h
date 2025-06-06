// emu.h
#ifndef EMU_H
#define EMU_H

#include <string>
#include <vector>
#include "inputhandler.h"
#include "../libgambatte/gambatte.h"

namespace gambatte {
    class InputGetter;
    class GB;
}

class Emulator {
public:
    Emulator(const std::string& biosPath, const std::string& romPath);
    ~Emulator();

    bool isInitialized() const;
    int stepFrame(gambatte::uint_least32_t* video, gambatte::uint_least32_t* audio, std::size_t &samples);
    void setInputGetter(gambatte::InputGetter* inputGetter) {
        gb->setInputGetter(inputGetter, nullptr);
    }
    bool loadState(const std::string& stateFile) {
        return gb->loadState(stateFile);
    }
    uint8_t readMemory(uint16_t address) const; // Added declaration

private:
    gambatte::GB* gb;
    bool initialized;
    std::vector<uint32_t> video;
    std::vector<unsigned> audio;
    static constexpr int width = 160;
    static constexpr int height = 144;
};

struct GBInputState {
    bool up = false;
    bool down = false;
    bool left = false;
    bool right = false;
    bool a = false;
    bool b = false;
    bool start = false;
    bool select = false;
};

#endif // EMU_H