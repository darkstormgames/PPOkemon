#include "environments/pokered/pokered_rewards.h"
#include "emulation/emu_gbc.h"
#include <algorithm>
#include <cmath>

PokeredRewards::PokeredRewards() 
{
}

PokeredRewards::~PokeredRewards() 
{
}

float PokeredRewards::CalculateReward(Emulator* emulator, const GameState& last_state)
{
    // Read current game state directly from emulator
    GameState current_state = ReadGameState(emulator);
    
    float total_reward = 0.0f;
    
    // Calculate all reward components
    total_reward += CalculateMovementReward(current_state, last_state);
    total_reward += CalculateBadgeReward(current_state, last_state);
    total_reward += CalculateMapExplorationReward(current_state, last_state);
    total_reward += CalculatePartyProgressReward(current_state, last_state);
    total_reward += CalculatePokedexReward(current_state, last_state);
    total_reward += CalculateEventFlagsReward(current_state, last_state);
    total_reward += CalculateBattleReward(current_state, last_state);
    total_reward += CalculateMoneyReward(current_state, last_state);
    total_reward += CalculateStagnationPenalty(current_state, last_state);
    
    return total_reward;
}

PokeredRewards::GameState PokeredRewards::ReadGameState(Emulator* emulator)
{
    GameState state;
    
    // Read basic player and game state
    state.player_x = emulator->readMemory(PLAYER_X_COORD_ADDR);
    state.player_y = emulator->readMemory(PLAYER_Y_COORD_ADDR);
    state.badges = emulator->readMemory(BADGES_ADDR);
    state.map_n = emulator->readMemory(MAP_N_ADDR);
    
    // Read party data
    uint8_t party_count = emulator->readMemory(PARTY_COUNT_ADDR);
    
    // Check if party data has changed using checksum
    uint16_t party_checksum = CalculatePartyChecksum(emulator, party_count);
    party_data_dirty_ = (party_checksum != last_party_checksum_) || 
                        (party_count != cached_party_count_);
    
    if (party_data_dirty_) {
        state.party_hp.clear();
        state.party_levels.clear();
        state.party_max_hp.clear();
        state.party_status.clear();
        
        // Reserve space for efficiency
        state.party_hp.reserve(6);
        state.party_levels.reserve(6);
        state.party_max_hp.reserve(6);
        state.party_status.reserve(6);
        
        for (uint8_t i = 0; i < party_count && i < 6; ++i) {
            uint16_t base_addr = PARTY_MON_1_SPECIES_ADDR + (i * PARTY_MON_STRUCT_SIZE);
            
            state.party_hp.push_back(ReadU16BE(emulator, base_addr + 1, base_addr + 2));
            state.party_status.push_back(emulator->readMemory(base_addr + 4));
            state.party_levels.push_back(emulator->readMemory(PARTY_MON_1_LEVEL_ADDR + (i * PARTY_MON_STRUCT_SIZE)));
            state.party_max_hp.push_back(ReadU16BE(emulator,
                PARTY_MON_1_MAX_HP_HIGH_BYTE_ADDR + (i * PARTY_MON_STRUCT_SIZE), 
                PARTY_MON_1_MAX_HP_LOW_BYTE_ADDR + (i * PARTY_MON_STRUCT_SIZE)));
        }
        
        last_party_checksum_ = party_checksum;
        cached_party_count_ = party_count;
    }
    
    // Determine when to update pokedex (optimized conditions)
    uint16_t current_enemy_hp = ReadU16BE(emulator, ENEMY_MON_1_HP_HIGH_BYTE_ADDR, ENEMY_MON_1_HP_LOW_BYTE_ADDR);
    pokedex_dirty_ = pokedex_dirty_ || (current_enemy_hp == 0); // Reset if enemy defeated
    
    // Read Pokedex data (with caching)
    if (pokedex_dirty_) {
        state.pokedex_caught = 0;
        // Optimized reading - process 4 bytes at a time
        for (uint16_t addr = POKEDEX_CAUGHT_START_ADDR; addr < POKEDEX_CAUGHT_START_ADDR + 19; addr += 4) {
            if (addr + 3 < POKEDEX_CAUGHT_START_ADDR + 19) {
                uint32_t four_bytes = emulator->readMemory(addr) |
                                     (emulator->readMemory(addr + 1) << 8) |
                                     (emulator->readMemory(addr + 2) << 16) |
                                     (emulator->readMemory(addr + 3) << 24);
                state.pokedex_caught += __builtin_popcount(four_bytes);
            } else {
                // Handle remaining bytes
                for (uint16_t a = addr; a < POKEDEX_CAUGHT_START_ADDR + 19; ++a) {
                    state.pokedex_caught += __builtin_popcount(emulator->readMemory(a));
                }
            }
        }
        
        cached_pokedex_caught_ = state.pokedex_caught;
        pokedex_dirty_ = false;
    } else {
        state.pokedex_caught = cached_pokedex_caught_;
    }
    
    // Read event flags (with caching)
    if (event_flags_dirty_) {
        state.event_flags_sum = 0;
        // Process 4 bytes at a time for better cache efficiency
        constexpr uint16_t flags_range = EVENT_FLAGS_END_ADDR - EVENT_FLAGS_START_ADDR;
        for (uint16_t offset = 0; offset < flags_range; offset += 4) {
            uint16_t addr = EVENT_FLAGS_START_ADDR + offset;
            if (offset + 3 < flags_range) {
                uint32_t four_bytes = emulator->readMemory(addr) |
                                     (emulator->readMemory(addr + 1) << 8) |
                                     (emulator->readMemory(addr + 2) << 16) |
                                     (emulator->readMemory(addr + 3) << 24);
                state.event_flags_sum += __builtin_popcount(four_bytes);
            } else {
                // Handle remaining bytes
                for (uint16_t a = addr; a < EVENT_FLAGS_END_ADDR; ++a) {
                    state.event_flags_sum += __builtin_popcount(emulator->readMemory(a));
                }
            }
        }
        
        cached_event_flags_sum_ = state.event_flags_sum;
        event_flags_dirty_ = false;
    } else {
        state.event_flags_sum = cached_event_flags_sum_;
    }
    
    // Read enemy data
    state.enemy_hp = current_enemy_hp;
    state.enemy_level = emulator->readMemory(ENEMY_MON_1_LEVEL_ADDR);
    
    // Read money
    uint8_t money_b1 = emulator->readMemory(PLAYER_MONEY_ADDR_BYTE_1);
    uint8_t money_b2 = emulator->readMemory(PLAYER_MONEY_ADDR_BYTE_2);
    uint8_t money_b3 = emulator->readMemory(PLAYER_MONEY_ADDR_BYTE_3);
    state.money = (money_b1 << 16) | (money_b2 << 8) | money_b3;
    
    return state;
}

void PokeredRewards::InitializeGameState(Emulator* emulator, GameState& state)
{
    state = ReadGameState(emulator);
    
    // Clear visited maps and add current map
    state.visited_maps.clear();
    state.visited_maps.insert(state.map_n);
    
    // Reset movement tracking
    state.steps_without_movement = 0;
    state.recent_positions.clear();
    state.recent_positions.insert({state.player_x, state.player_y});
    state.last_position = {state.player_x, state.player_y};
    
    // Mark caches as needing update
    party_data_dirty_ = true;
    pokedex_dirty_ = true;
    event_flags_dirty_ = true;
}

void PokeredRewards::UpdateMovementTracking(GameState& current_state, const GameState& last_state, uint32_t episode_length)
{
    // Update movement tracking
    std::pair<uint8_t, uint8_t> current_pos = {current_state.player_x, current_state.player_y};
    
    if (current_pos != current_state.last_position) {
        current_state.steps_without_movement = 0;
        current_state.recent_positions.insert(current_pos);
        
        // Limit the size of recent positions
        if (current_state.recent_positions.size() > RECENT_POSITIONS_WINDOW) {
            // Remove oldest positions (this is a simplification; in practice, you'd want to track timestamps)
            auto it = current_state.recent_positions.begin();
            std::advance(it, current_state.recent_positions.size() - RECENT_POSITIONS_WINDOW);
            current_state.recent_positions.erase(current_state.recent_positions.begin(), it);
        }
        
        current_state.last_position = current_pos;
    } else {
        current_state.steps_without_movement++;
    }
    
    // Update visited maps
    current_state.visited_maps.insert(current_state.map_n);
    
    // Reset recent positions periodically to encourage exploration
    if (episode_length % 4096 == 0) {
        current_state.recent_positions.clear();
        current_state.recent_positions.insert(current_pos);
    }
}

uint16_t PokeredRewards::ReadU16BE(Emulator* emulator, uint16_t high_addr, uint16_t low_addr) const
{
    return (static_cast<uint16_t>(emulator->readMemory(high_addr)) << 8) | emulator->readMemory(low_addr);
}

uint16_t PokeredRewards::CalculatePartyChecksum(Emulator* emulator, uint8_t party_count) const
{
    uint16_t checksum = party_count;
    for (uint8_t i = 0; i < party_count && i < 6; ++i) {
        uint16_t base_addr = PARTY_MON_1_SPECIES_ADDR + (i * PARTY_MON_STRUCT_SIZE);
        checksum ^= emulator->readMemory(base_addr); // Species
        checksum = (checksum << 1) | (checksum >> 15); // Rotate left
        checksum ^= emulator->readMemory(PARTY_MON_1_LEVEL_ADDR + (i * PARTY_MON_STRUCT_SIZE)); // Level
    }
    return checksum;
}

float PokeredRewards::CalculateMovementReward(const GameState& current_state, 
                                             const GameState& last_state)
{
    // Small reward for movement to encourage exploration
    if (current_state.player_x != last_state.player_x || 
        current_state.player_y != last_state.player_y) {
        
        // Check if this position was recently visited
        std::pair<uint8_t, uint8_t> current_pos = {current_state.player_x, current_state.player_y};
        if (current_state.recent_positions.find(current_pos) == current_state.recent_positions.end()) {
            // New position not recently visited - give exploration reward
            return MOVEMENT_REWARD_SCALE;
        } else {
            // Position was recently visited - give small penalty
            return MOVEMENT_REWARD_SCALE * -0.5f;
        }
    }
    return 0.0f;
}

float PokeredRewards::CalculateBadgeReward(const GameState& current_state, 
                                          const GameState& last_state)
{
    // Large reward for collecting badges
    int badge_diff = __builtin_popcount(current_state.badges) - 
                     __builtin_popcount(last_state.badges);
    return badge_diff * BADGE_REWARD_SCALE;
}

float PokeredRewards::CalculateMapExplorationReward(const GameState& current_state, 
                                                   const GameState& last_state)
{
    // Reward for visiting new maps
    if (current_state.visited_maps.size() > last_state.visited_maps.size()) {
        return (current_state.visited_maps.size() - last_state.visited_maps.size()) * MAP_EXPLORATION_SCALE;
    }
    return 0.0f;
}

float PokeredRewards::CalculatePartyProgressReward(const GameState& current_state, 
                                                  const GameState& last_state)
{
    float party_reward = 0.0f;
    
    // Reward for Pokemon level ups
    size_t min_count = std::min(current_state.party_levels.size(), last_state.party_levels.size());
    for (size_t i = 0; i < min_count; ++i) {
        int level_diff = current_state.party_levels[i] - last_state.party_levels[i];
        if (level_diff > 0) {
            party_reward += level_diff * PARTY_PROGRESS_SCALE;
        }
    }
    
    // Reward for healing Pokemon (HP increase) and penalize fainting
    for (size_t i = 0; i < min_count; ++i) {
        if (i < current_state.party_hp.size() && i < last_state.party_hp.size() &&
            i < current_state.party_max_hp.size() && i < last_state.party_max_hp.size()) {
            
            // HP change as fraction of max HP
            if (last_state.party_max_hp[i] > 0) {
                float hp_change = static_cast<float>(current_state.party_hp[i] - last_state.party_hp[i]);
                float hp_change_fraction = hp_change / static_cast<float>(last_state.party_max_hp[i]);
                party_reward += hp_change_fraction * PARTY_PROGRESS_SCALE * 0.05f;
            }
            
            // Fainting penalty
            if (current_state.party_hp[i] == 0 && last_state.party_hp[i] > 0) {
                party_reward -= 0.25f;
            }
        }
    }
    
    return party_reward;
}

float PokeredRewards::CalculatePokedexReward(const GameState& current_state, 
                                            const GameState& last_state)
{
    // Reward for catching new Pokemon
    int pokedex_diff = current_state.pokedex_caught - last_state.pokedex_caught;
    return pokedex_diff * POKEDEX_REWARD_SCALE;
}

float PokeredRewards::CalculateEventFlagsReward(const GameState& current_state, 
                                               const GameState& last_state)
{
    // Small reward for triggering events (story progress)
    int event_diff = current_state.event_flags_sum - last_state.event_flags_sum;
    return event_diff * EVENT_FLAGS_SCALE;
}

float PokeredRewards::CalculateBattleReward(const GameState& current_state, 
                                           const GameState& last_state)
{
    // Reward for battle progress (enemy Pokemon taking damage)
    if (last_state.enemy_hp > current_state.enemy_hp && current_state.enemy_hp > 0) {
        int damage_dealt = last_state.enemy_hp - current_state.enemy_hp;
        return damage_dealt * BATTLE_REWARD_SCALE * 0.01f;
    }
    
    // Large reward for defeating enemy Pokemon
    if (last_state.enemy_hp > 0 && current_state.enemy_hp == 0) {
        return BATTLE_REWARD_SCALE;
    }
    
    return 0.0f;
}

float PokeredRewards::CalculateMoneyReward(const GameState& current_state, 
                                          const GameState& last_state)
{
    // Small reward for gaining money
    if (current_state.money > last_state.money) {
        return (current_state.money - last_state.money) * MONEY_REWARD_SCALE;
    }
    return 0.0f;
}

float PokeredRewards::CalculateStagnationPenalty(const GameState& current_state, 
                                                 const GameState& last_state)
{
    // Penalty for staying in the same area too long without progress
    if (current_state.steps_without_movement > MAX_STEPS_WITHOUT_MOVEMENT) {
        return STAGNATION_PENALTY_SCALE * (current_state.steps_without_movement - MAX_STEPS_WITHOUT_MOVEMENT);
    }
    
    // Penalty for cycling through too few positions
    if (current_state.recent_positions.size() < 5 && current_state.steps_without_movement > 50) {
        return STAGNATION_PENALTY_SCALE * 10.0f;
    }
    
    // Apply increasing penalty for staying in the same place
    if (current_state.steps_without_movement > 60) {  // After ~1 second of no movement
        float penalty = std::min(0.01f, current_state.steps_without_movement * 0.00001f);
        return -penalty;
    }
    
    return 0.0f;
}