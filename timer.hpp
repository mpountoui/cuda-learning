#include <chrono>
#include <string>

/*------------------------------------------------------------------------------------------*/

class Timer
{
public:
    explicit Timer(const std::string& name);
    
    void start();
    void stop();
    void reset();
    
    double elapsedSeconds() const;
    long long elapsedMilliseconds() const;
    long long elapsedMicroseconds() const;
    
private:
    using clock = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;
    
    std::string m_name;
    time_point m_start;
    time_point m_end;
    bool m_running = false;
};
