#include "usec.h"
#include <chrono>
#include <thread>

usec_t getusecs() {
    using namespace std::chrono;
    return duration_cast<microseconds>(
               steady_clock::now().time_since_epoch())
        .count();
}

void usecsleep(usec_t usecs) {
    std::this_thread::sleep_for(std::chrono::microseconds(usecs));
}