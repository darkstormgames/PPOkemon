// Example of using the new Pokered environment with separated rewards
// This demonstrates the clean separation between environment logic and reward calculation

#include "environments/pokered/pokered.h"
#include "environments/pokered/pokered_rewards.h"
#include <iostream>
#include <vector>

int main() {
    try {
        // Create environment with specified render mode
        Pokered env(RenderMode::Headless, 42);
        
        std::cout << "Pokemon Red Environment Initialized!" << std::endl;
        std::cout << "Observation size: " << env.GetObservationSize() << std::endl;
        std::cout << "Action size: " << env.GetActionSize() << std::endl;
        
        // Reset environment
        env.Reset();
        
        // Get initial observation
        std::vector<float> observation(env.GetObservationSize());
        env.GetObsData(observation.data());
        
        // Example: Take some random actions
        for (int step = 0; step < 10; ++step) {
            // Create random action (8 buttons)
            std::vector<float> action(env.GetActionSize(), 0.0f);
            
            // Simulate pressing a button (right = 0, left = 1, up = 2, down = 3, A = 4, B = 5, start = 6, select = 7)
            action[step % env.GetActionSize()] = 1.0f;
            
            // Step environment
            auto result = env.Step(action.data(), action.size());
            
            // Display game state information
            const auto& current_state = env.GetCurrentGameState();
            std::cout << "Step " << step + 1 << ":" << std::endl;
            std::cout << "  Player position: (" << static_cast<int>(current_state.player_x) 
                      << ", " << static_cast<int>(current_state.player_y) << ")" << std::endl;
            std::cout << "  Map: " << static_cast<int>(current_state.map_n) << std::endl;
            std::cout << "  Badges: " << __builtin_popcount(current_state.badges) << std::endl;
            std::cout << "  Reward: " << result.reward << std::endl;
            std::cout << "  Money: " << current_state.money << std::endl;
            
            // Get updated observation
            env.GetObsData(observation.data());
            
            if (result.done) {
                std::cout << "Episode finished!" << std::endl;
                break;
            }
        }
        
        std::cout << "Example completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
