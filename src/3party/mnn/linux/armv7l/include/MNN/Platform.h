#pragma once

#include <mutex>
#if defined(AMBA_H22)

#include <string>

#define AMBA_H22_NR_AVAILABLE_CORES 4

namespace std {

class mutex {   // note: If you want to run multiple sessions on different threads on amba h22, you should implement the mutex
public:
    void lock(){};
    void unlock(){};
};

struct once_flag {
    volatile bool called = false;
};

template< class Callable, class... Args >
void call_once(std::once_flag& flag, Callable&& f, Args&&... args ) {
    if (flag.called == true) {
        return;
    }
    
    flag.called = true;
    
    f(args...);
}

}
#endif