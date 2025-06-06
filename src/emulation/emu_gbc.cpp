// emu.cpp
#include "emu_gbc.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <libgambatte/gbint.h>

Emulator::Emulator(const std::string &biosPath, const std::string &romPath)
    : initialized(false), video(160 * 144), audio(37000 + 2064)
{

    // std::cout << "BIOS Path: " << biosPath << "\n";
    // std::cout << "ROM Path: " << romPath << "\n";

    gb = new gambatte::GB();
    if (!gb)
    {
        std::cerr << "Failed to create Gambatte instance.\n";
        return;
    }

    if (!std::filesystem::exists(biosPath) || !std::filesystem::exists(romPath))
    {
        std::cerr << "BIOS or ROM path does not exist.\n";
        return;
    }

    auto readFile = [](const std::string &path) -> std::vector<char>
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file)
            return {};
        auto size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        file.read(buffer.data(), size);
        return buffer;
    };

    auto biosData = readFile(biosPath);
    int result = gb->loadBios(biosData.data(), biosData.size());
    if (result != 0)
    {
        std::cerr << "Failed to load BIOS: " << biosPath << " (" << result << ")\n";
        return;
    }

    auto romData = readFile(romPath);
    result = gb->load(romData.data(), romData.size(), gambatte::GB::CGB_MODE | gambatte::GB::READONLY_SAV);
    if (result != 0)
    {
        std::cerr << "Failed to load ROM: " << romPath << " (" << result << ")\n";
        return;
    }

    gb->setSaveDir(std::filesystem::path(romPath).parent_path().string());

    gb->reset(0, ""); // Reset with no samples to stall and no specific build

    // Extract ROM name without extension
    std::string romName = std::filesystem::path(romPath).stem().string();
    std::string stateFile = std::filesystem::path(romPath).parent_path().string() + "/" + romName + "_1.gqs";
    gb->loadState(stateFile.c_str());

    initialized = true;
}

Emulator::~Emulator()
{
    if (gb)
        delete gb;
}

bool Emulator::isInitialized() const
{
    return initialized;
}

int Emulator::stepFrame(gambatte::uint_least32_t *video, gambatte::uint_least32_t *audio, std::size_t &samples)
{
    if (!gb)
        return -1;
    return gb->runFor(video, 160, audio, samples);
}

// Implementation of readMemory
uint8_t Emulator::readMemory(uint16_t address) const
{
    if (!gb)
    {
        std::cerr << "Error: Gambatte instance not initialized in readMemory." << std::endl;
        return 0xFF;
    }

    unsigned char *data_ptr = nullptr;
    int length = 0;
    bool success = false;
    uint16_t offset_in_area = 0;
    int memory_area_type = -1;

    // Map address to memory area type and calculate offset within that area
    // Based on Gambatte's getMemoryArea 'which' parameter:
    // 0 = vram, 1 = rom, 2 = wram, 3 = cartram, 4 = oam, 5 = hram
    if (address <= 0x7FFF)
    {                             // ROM
        memory_area_type = 1;     // ROM
        offset_in_area = address; // Offset is the address itself relative to the start of mapped ROM
    }
    else if (address >= 0x8000 && address <= 0x9FFF)
    {                         // VRAM
        memory_area_type = 0; // VRAM
        offset_in_area = address - 0x8000;
    }
    else if (address >= 0xA000 && address <= 0xBFFF)
    {                         // Cartridge RAM
        memory_area_type = 3; // CARTRAM
        offset_in_area = address - 0xA000;
    }
    else if (address >= 0xC000 && address <= 0xDFFF)
    {                         // WRAM
        memory_area_type = 2; // WRAM
        offset_in_area = address - 0xC000;
    }
    else if (address >= 0xE000 && address <= 0xFDFF)
    {                                      // Echo RAM (mirror of C000-DDFF WRAM)
        memory_area_type = 2;              // WRAM
        offset_in_area = address - 0xE000; // Offset relative to the start of the C000-DDFF WRAM block
    }
    else if (address >= 0xFE00 && address <= 0xFE9F)
    {                         // OAM
        memory_area_type = 4; // OAM
        offset_in_area = address - 0xFE00;
    }
    else if (address >= 0xFF80 && address <= 0xFFFE)
    {                         // HRAM
        memory_area_type = 5; // HRAM
        offset_in_area = address - 0xFF80;
    }
    else
    {
        // Addresses not covered by getMemoryArea's defined types
        // (e.g., I/O registers FF00-FF7F, Unusable FEA0-FEFF, IE FFFF)
        // gb->externalRead(address) could be used here if direct hardware register access is needed.
        // For now, sticking to getMemoryArea as requested.
        std::cerr << "Warning: Address 0x" << std::hex << address << std::dec
                  << " corresponds to a memory region not directly accessible via getMemoryArea's current mapping. Returning 0xFF." << std::endl;
        return 0xFF;
    }

    // Ensure a ROM is loaded before attempting to get memory areas
    if (!gb->isLoaded())
    {
        std::cerr << "Error: No ROM loaded. Cannot access memory area for address 0x" << std::hex << address << std::dec << "." << std::endl;
        return 0xFF;
    }

    success = gb->getMemoryArea(memory_area_type, &data_ptr, &length);

    if (success && data_ptr != nullptr)
    {
        if (offset_in_area < static_cast<uint16_t>(length))
        {
            return data_ptr[offset_in_area];
        }
        else
        {
            std::cerr << "Error: Calculated offset 0x" << std::hex << offset_in_area << std::dec
                      << " is out of bounds for memory area type " << memory_area_type
                      << " (length: " << length << ", accessed via address 0x" << std::hex << address << std::dec
                      << "). Returning 0xFF." << std::endl;
            return 0xFF;
        }
    }
    else
    {
        std::cerr << "Error: gb->getMemoryArea failed or returned null pointer for memory area type " << memory_area_type
                  << " (accessed via address 0x" << std::hex << address << std::dec << ", success: " << success
                  << ", data_ptr: " << static_cast<void *>(data_ptr) << "). Returning 0xFF." << std::endl;
        return 0xFF;
    }
}
