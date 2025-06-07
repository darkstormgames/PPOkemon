#pragma once

#include <cstdint>
#include <vector>
#include <set>
#include <memory>

// Forward declarations
class Emulator;

class PokeredRewards 
{
public:
    // Lightweight game state structure for rewards
    struct GameState {
        // Player and map data
        uint8_t player_x;
        uint8_t player_y;
        uint8_t badges;
        uint8_t map_n;
        
        // Party tracking
        std::vector<uint16_t> party_hp;
        std::vector<uint8_t> party_levels;
        std::vector<uint16_t> party_max_hp;
        std::vector<uint8_t> party_status;
        
        // Game progress
        uint16_t pokedex_caught;
        uint32_t event_flags_sum;
        uint16_t enemy_hp;
        uint8_t enemy_level;
        uint32_t money;
        
        // Movement tracking
        std::set<uint8_t> visited_maps;
        uint32_t steps_without_movement;
        std::set<std::pair<uint8_t, uint8_t>> recent_positions;
        std::pair<uint8_t, uint8_t> last_position;
        
        // Interactive state tracking
        bool is_in_interactive_state;
        uint32_t steps_in_battle;
        uint32_t steps_in_dialog;
        uint32_t steps_in_menu;
        bool enemy_present_in_battle;
    };

    PokeredRewards();
    ~PokeredRewards();

    // Main reward calculation function - now takes emulator directly
    float CalculateReward(Emulator* emulator, const GameState& last_state);
    
    // State management functions
    GameState ReadGameState(Emulator* emulator);
    void InitializeGameState(Emulator* emulator, GameState& state);
    void UpdateMovementTracking(GameState& current_state, const GameState& last_state, uint32_t episode_length, Emulator* emulator);

    // Game state detection helpers (public for testing)
    bool IsInInteractiveState(Emulator* emulator) const;
    bool IsInBattle(Emulator* emulator) const;
    bool IsInDialog(Emulator* emulator) const;
    bool IsInMenu(Emulator* emulator) const;

    // For unit testing - make penalty calculation public
    float CalculateStagnationPenalty(const GameState& current_state, 
                                    const GameState& last_state);

    // Memory address constants (public for testing)
    static constexpr uint16_t W_IS_IN_BATTLE_ADDR = 0xD057;        // wIsInBattle
    static constexpr uint16_t W_TEXT_BOX_ID_ADDR = 0xD125;        // wTextBoxID
    static constexpr uint16_t W_JOY_IGNORE_ADDR = 0xCD6B;         // wJoyIgnore - non-zero when input disabled
    static constexpr uint16_t W_CURRENT_MENU_ITEM_ADDR = 0xCC26;  // wCurrentMenuItem
    static constexpr uint16_t W_MAX_MENU_ITEM_ADDR = 0xCC28;      // wMaxMenuItem - indicates menu is open

private:
    // Individual reward components
    float CalculateMovementReward(const GameState& current_state, 
                                 const GameState& last_state);
    
    float CalculateBadgeReward(const GameState& current_state, 
                              const GameState& last_state);
    
    float CalculateMapExplorationReward(const GameState& current_state, 
                                       const GameState& last_state);
    
    float CalculatePartyProgressReward(const GameState& current_state, 
                                      const GameState& last_state);
    
    float CalculatePokedexReward(const GameState& current_state, 
                                const GameState& last_state);
    
    float CalculateEventFlagsReward(const GameState& current_state, 
                                   const GameState& last_state);
    
    float CalculateBattleReward(const GameState& current_state, 
                               const GameState& last_state);
    
    float CalculateMoneyReward(const GameState& current_state, 
                              const GameState& last_state);

    // Memory reading helpers
    uint16_t ReadU16BE(Emulator* emulator, uint16_t high_addr, uint16_t low_addr) const;
    uint16_t CalculatePartyChecksum(Emulator* emulator, uint8_t party_count) const;
    
    // Reward scaling constants
    static constexpr float MOVEMENT_REWARD_SCALE = 0.001f;
    static constexpr float BADGE_REWARD_SCALE = 1.0f;
    static constexpr float MAP_EXPLORATION_SCALE = 0.1f;
    static constexpr float PARTY_PROGRESS_SCALE = 0.2f;
    static constexpr float POKEDEX_REWARD_SCALE = 0.3f;
    static constexpr float EVENT_FLAGS_SCALE = 0.5f;
    static constexpr float BATTLE_REWARD_SCALE = 0.5f;
    static constexpr float MONEY_REWARD_SCALE = 0.0001f;
    static constexpr float STAGNATION_PENALTY_SCALE = -0.1f;
    static constexpr uint32_t MAX_STEPS_WITHOUT_MOVEMENT = 1024;
    static constexpr uint32_t RECENT_POSITIONS_WINDOW = 100;

    // Memory addresses - now private to rewards system
    static constexpr uint16_t PLAYER_X_COORD_ADDR = 0xD362;
    static constexpr uint16_t PLAYER_Y_COORD_ADDR = 0xD361;
    static constexpr uint16_t MAP_N_ADDR = 0xD35E;
    static constexpr uint16_t BADGES_ADDR = 0xD356;
    
    static constexpr uint16_t PARTY_COUNT_ADDR = 0xD163;
    static constexpr uint16_t PARTY_MON_STRUCT_SIZE = 44;
    static constexpr uint16_t PARTY_MON_1_SPECIES_ADDR = 0xD16B;
    static constexpr uint16_t PARTY_MON_1_HP_HIGH_BYTE_ADDR = 0xD16C;
    static constexpr uint16_t PARTY_MON_1_HP_LOW_BYTE_ADDR = 0xD16D;
    static constexpr uint16_t PARTY_MON_1_STATUS_ADDR = 0xD16F;
    static constexpr uint16_t PARTY_MON_1_LEVEL_ADDR = 0xD18C;
    static constexpr uint16_t PARTY_MON_1_MAX_HP_HIGH_BYTE_ADDR = 0xD18D;
    static constexpr uint16_t PARTY_MON_1_MAX_HP_LOW_BYTE_ADDR = 0xD18E;
    
    static constexpr uint16_t POKEDEX_CAUGHT_START_ADDR = 0xD2F7;
    static constexpr uint16_t POKEDEX_OWNED_START_ADDR = 0xD30A;
    
    static constexpr uint16_t EVENT_FLAGS_START_ADDR = 0xD747;
    static constexpr uint16_t EVENT_FLAGS_END_ADDR = 0xD886;
    
    static constexpr uint16_t ENEMY_MON_1_LEVEL_ADDR = 0xD8C5;
    static constexpr uint16_t ENEMY_MON_1_HP_HIGH_BYTE_ADDR = 0xD8A5;
    static constexpr uint16_t ENEMY_MON_1_HP_LOW_BYTE_ADDR = 0xD8A6;
    
    static constexpr uint16_t PLAYER_MONEY_ADDR_BYTE_1 = 0xD347;
    static constexpr uint16_t PLAYER_MONEY_ADDR_BYTE_2 = 0xD348;
    static constexpr uint16_t PLAYER_MONEY_ADDR_BYTE_3 = 0xD349;

    // Cache control for optimized memory reading
    mutable bool party_data_dirty_ = true;
    mutable uint8_t cached_party_count_ = 0;
    mutable bool pokedex_dirty_ = true;
    mutable bool event_flags_dirty_ = true;
    mutable uint16_t cached_pokedex_caught_ = 0;
    mutable uint32_t cached_event_flags_sum_ = 0;
    mutable uint16_t last_party_checksum_ = 0;
};
