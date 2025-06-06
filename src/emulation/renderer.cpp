// renderer.cpp
#include "renderer.h"

Renderer::Renderer(int w, int h, int s, RenderMode m)
    : width(w), height(h), scale(s), mode(m)
{
    if (mode == RenderMode::SDL)
    {
#ifndef SDL_AVAILABLE
        std::cerr << "SDL not available in this build.\n";
        return;
#else
        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO) != 0)
        {
            std::cerr << "SDL_Init Error: " << SDL_GetError() << "\n";
            return;
        }

        window = SDL_CreateWindow("Gambatte SDL", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                  width * scale, height * scale, SDL_WINDOW_SHOWN);
        if (!window)
        {
            std::cerr << "SDL_CreateWindow Error: " << SDL_GetError() << "\n";
            return;
        }

        sdlRenderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        if (!sdlRenderer)
        {
            std::cerr << "SDL_CreateRenderer Error: " << SDL_GetError() << "\n";
            return;
        }

        texture = SDL_CreateTexture(sdlRenderer, SDL_PIXELFORMAT_ARGB8888,
                                    SDL_TEXTUREACCESS_STREAMING, width, height);
        if (!texture)
        {
            std::cerr << "SDL_CreateTexture Error: " << SDL_GetError() << "\n";
            return;
        }
#endif
    }

    initialized = true;
}

Renderer::~Renderer()
{
#ifdef SDL_AVAILABLE
    if (mode == RenderMode::SDL)
    {
        if (texture)
            SDL_DestroyTexture(texture);
        if (sdlRenderer)
            SDL_DestroyRenderer(sdlRenderer);
        if (window)
            SDL_DestroyWindow(window);
        SDL_Quit();
    }
#endif
}

bool Renderer::isInitialized() const
{
    return initialized;
}

void Renderer::renderFrame(const unsigned int *pixels)
{
    if (!initialized || mode == RenderMode::Headless)
        return;

    switch (mode)
    {
    case RenderMode::SDL:
#ifdef SDL_AVAILABLE
        if (SDL_UpdateTexture(texture, nullptr, pixels, width * sizeof(uint32_t)) != 0)
        {
            std::cerr << "SDL_UpdateTexture Error: " << SDL_GetError() << "\n";
        }
        SDL_RenderClear(sdlRenderer);
        SDL_RenderCopy(sdlRenderer, texture, nullptr, nullptr);
        SDL_RenderPresent(sdlRenderer);
#endif
        break;
    case RenderMode::Console:
        std::cout << "\033[H";
        renderFrameToConsole(pixels);
        break;

    case RenderMode::Headless:
        break; // No rendering in headless mode
    }
}

void Renderer::renderFrameToConsole(const unsigned int *video)
{
    const int blockSize = consoleBlockSize;
    for (int y = 0; y < height; y += blockSize)
    {
        std::cout << "\033[0G";
        for (int x = 0; x < width; x += blockSize)
        {
            int rSum = 0, gSum = 0, bSum = 0, count = 0;

            for (int dy = 0; dy < blockSize; ++dy)
            {
                for (int dx = 0; dx < blockSize; ++dx)
                {
                    int px = x + dx;
                    int py = y + dy;
                    if (px < width && py < height)
                    {
                        uint32_t color = video[py * width + px];
                        uint8_t r = (color >> 16) & 0xFF;
                        uint8_t g = (color >> 8) & 0xFF;
                        uint8_t b = color & 0xFF;
                        rSum += r;
                        gSum += g;
                        bSum += b;
                        count++;
                    }
                }
            }

            uint8_t rAvg = rSum / count;
            uint8_t gAvg = gSum / count;
            uint8_t bAvg = bSum / count;

            std::cout << "\033[38;2;" << (int)rAvg << ";" << (int)gAvg << ";" << (int)bAvg << "m██";
        }
        std::cout << std::endl;
    }
}