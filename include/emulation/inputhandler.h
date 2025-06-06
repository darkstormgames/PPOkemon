// inputhandler.h
#ifndef INPUTHANDLER_H
#define INPUTHANDLER_H

#include <atomic>
#include "emu_gbc.h"
#include "renderer.h"
#include "../libgambatte/inputgetter.h"

// SDL conditional inclusion
#if __has_include(<SDL2/SDL.h>)
#include <SDL2/SDL.h>
#define SDL_AVAILABLE
#endif

class InputHandler : public gambatte::InputGetter
{
public:
    explicit InputHandler(RenderMode mode = RenderMode::Headless);
    ~InputHandler();

    void poll(std::atomic<bool> &running);
    void setInputState(unsigned newState);

    // From InputGetter
    unsigned operator()() override;

private:
    struct InputState
    {
        bool a = false, b = false, select = false, start = false;
        bool up = false, down = false, left = false, right = false;
    } currentState;

    enum class InputSource
    {
        None,
        Manual,
        SDL
    };

    struct SourceState
    {
        InputSource a = InputSource::None, b = InputSource::None;
        InputSource select = InputSource::None, start = InputSource::None;
        InputSource up = InputSource::None, down = InputSource::None;
        InputSource left = InputSource::None, right = InputSource::None;
    };

    SourceState sourceTracker;

    InputState defaultState;
    InputState sdlOverride;

    unsigned lastResolvedInput = 0x00;

#ifdef SDL_AVAILABLE
    bool sdlInitialized;
#endif
};

#endif // INPUTHANDLER_H