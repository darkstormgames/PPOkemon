// renderer.h
#pragma once

#include <vector>
#include <cstdint>
#include <iostream>
#include "../torch_rl/utils/args.h"  // Include args.h for RenderMode

#ifdef USE_SDL2
#include <SDL2/SDL.h>
#define SDL_AVAILABLE
#endif

class Renderer {
public:
    Renderer(int width, int height, int scale, RenderMode mode);
    ~Renderer();

    bool isInitialized() const;
    void renderFrame(const unsigned int* pixels);
    void setConsoleBlockSize(unsigned int blockSize) {
        consoleBlockSize = blockSize;
    }

private:
    void renderFrameToConsole(const unsigned int* video);

    int width, height, scale;
    int consoleBlockSize = 2; // Default block size for console rendering
    RenderMode mode;
    bool initialized = false;

#ifdef SDL_AVAILABLE
    SDL_Window* window = nullptr;
    SDL_Renderer* sdlRenderer = nullptr;
    SDL_Texture* texture = nullptr;
#endif
};