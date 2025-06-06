// inputhandler.cpp
#include "inputhandler.h"

#ifdef USE_SDL2
InputHandler::InputHandler(RenderMode mode)
    : sdlInitialized(false)
{
    if (mode == RenderMode::SDL)
    {
        sdlInitialized = SDL_Init(SDL_INIT_EVENTS) == 0;
    }
}
InputHandler::~InputHandler()
{
    if (sdlInitialized)
        SDL_QuitSubSystem(SDL_INIT_EVENTS);
}
#else
InputHandler::InputHandler(RenderMode) {}
InputHandler::~InputHandler() {}
#endif

void InputHandler::poll(std::atomic<bool> &running)
{
#ifdef USE_SDL2
    if (!sdlInitialized)
        return;

    // sdlOverride = {}; // clear all overrides before polling

    SDL_Event e;
    while (SDL_PollEvent(&e))
    {
        if (e.type == SDL_QUIT)
        {
            running = false;
        }
        else if (e.type == SDL_KEYDOWN || e.type == SDL_KEYUP)
        {
            bool pressed = (e.type == SDL_KEYDOWN);
            switch (e.key.keysym.sym)
            {
            case SDLK_UP:
                sdlOverride.up = pressed;
                sourceTracker.up = InputSource::SDL;
                break;
            case SDLK_DOWN:
                sdlOverride.down = pressed;
                sourceTracker.down = InputSource::SDL;
                break;
            case SDLK_LEFT:
                sdlOverride.left = pressed;
                sourceTracker.left = InputSource::SDL;
                break;
            case SDLK_RIGHT:
                sdlOverride.right = pressed;
                sourceTracker.right = InputSource::SDL;
                break;
            case SDLK_c:
                sdlOverride.a = pressed;
                sourceTracker.a = InputSource::SDL;
                break;
            case SDLK_x:
                sdlOverride.b = pressed;
                sourceTracker.b = InputSource::SDL;
                break;
            case SDLK_RETURN:
                sdlOverride.start = pressed;
                sourceTracker.start = InputSource::SDL;
                break;
            case SDLK_RSHIFT:
                sdlOverride.select = pressed;
                sourceTracker.select = InputSource::SDL;
                break;
            }
        }
    }
#else
    (void)running;
#endif
}

void InputHandler::setInputState(unsigned newState)
{
    defaultState.a = (newState & gambatte::InputGetter::A) != 0;
    sourceTracker.a = InputSource::Manual;
    defaultState.b = (newState & gambatte::InputGetter::B) != 0;
    sourceTracker.b = InputSource::Manual;
    defaultState.select = (newState & gambatte::InputGetter::SELECT) != 0;
    sourceTracker.select = InputSource::Manual;
    defaultState.start = (newState & gambatte::InputGetter::START) != 0;
    sourceTracker.start = InputSource::Manual;
    defaultState.right = (newState & gambatte::InputGetter::RIGHT) != 0;
    sourceTracker.right = InputSource::Manual;
    defaultState.left = (newState & gambatte::InputGetter::LEFT) != 0;
    sourceTracker.left = InputSource::Manual;
    defaultState.up = (newState & gambatte::InputGetter::UP) != 0;
    sourceTracker.up = InputSource::Manual;
    defaultState.down = (newState & gambatte::InputGetter::DOWN) != 0;
    sourceTracker.down = InputSource::Manual;
}

static const std::pair<unsigned, const char *> buttonMap[] = {
    {gambatte::InputGetter::A, "A"},
    {gambatte::InputGetter::B, "B"},
    {gambatte::InputGetter::SELECT, "Select"},
    {gambatte::InputGetter::START, "Start"},
    {gambatte::InputGetter::RIGHT, "Right"},
    {gambatte::InputGetter::LEFT, "Left"},
    {gambatte::InputGetter::UP, "Up"},
    {gambatte::InputGetter::DOWN, "Down"}};

unsigned InputHandler::operator()()
{
    auto isPressed = [](bool manual, bool override)
    {
        return override ? override : manual;
    };

    unsigned input = 0x00;

    if (isPressed(defaultState.a, sdlOverride.a))
        input |= gambatte::InputGetter::A;
    if (isPressed(defaultState.b, sdlOverride.b))
        input |= gambatte::InputGetter::B;
    if (isPressed(defaultState.select, sdlOverride.select))
        input |= gambatte::InputGetter::SELECT;
    if (isPressed(defaultState.start, sdlOverride.start))
        input |= gambatte::InputGetter::START;
    if (isPressed(defaultState.right, sdlOverride.right))
        input |= gambatte::InputGetter::RIGHT;
    if (isPressed(defaultState.left, sdlOverride.left))
        input |= gambatte::InputGetter::LEFT;
    if (isPressed(defaultState.up, sdlOverride.up))
        input |= gambatte::InputGetter::UP;
    if (isPressed(defaultState.down, sdlOverride.down))
        input |= gambatte::InputGetter::DOWN;

    // Logging if debugging
    #ifdef BUILD_DEBUG
    // if (input != lastResolvedInput)
    // {
    //     std::vector<std::string> changes;
    //     for (const auto &[bit, name] : buttonMap)
    //     {
    //         bool prevPressed = lastResolvedInput & bit;
    //         bool currPressed = input & bit;
    //         if (prevPressed != currPressed)
    //         {
    //             std::string src;
    //             switch (bit)
    //             {
    //             case gambatte::InputGetter::A:
    //                 src = (sourceTracker.a == InputSource::SDL ? "SDL" : "Manual");
    //                 break;
    //             case gambatte::InputGetter::B:
    //                 src = (sourceTracker.b == InputSource::SDL ? "SDL" : "Manual");
    //                 break;
    //             case gambatte::InputGetter::SELECT:
    //                 src = (sourceTracker.select == InputSource::SDL ? "SDL" : "Manual");
    //                 break;
    //             case gambatte::InputGetter::START:
    //                 src = (sourceTracker.start == InputSource::SDL ? "SDL" : "Manual");
    //                 break;
    //             case gambatte::InputGetter::RIGHT:
    //                 src = (sourceTracker.right == InputSource::SDL ? "SDL" : "Manual");
    //                 break;
    //             case gambatte::InputGetter::LEFT:
    //                 src = (sourceTracker.left == InputSource::SDL ? "SDL" : "Manual");
    //                 break;
    //             case gambatte::InputGetter::UP:
    //                 src = (sourceTracker.up == InputSource::SDL ? "SDL" : "Manual");
    //                 break;
    //             case gambatte::InputGetter::DOWN:
    //                 src = (sourceTracker.down == InputSource::SDL ? "SDL" : "Manual");
    //                 break;
    //             }

    //             changes.emplace_back(std::string(name) + " = " + (currPressed ? "pressed" : "released") + " (" + src + ")");
    //         }
    //     }

    //     if (!changes.empty())
    //     {
    //         std::cout << "Input changed: ";
    //         for (const auto &change : changes)
    //             std::cout << change << " ";
    //         std::cout << std::endl;
    //     }

    //     lastResolvedInput = input;
    // }
    #endif // BUILD_DEBUG

    return input;
}