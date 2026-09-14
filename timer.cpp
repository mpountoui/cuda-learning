#include "timer.hpp"
#include <iostream>

/*------------------------------------------------------------------------------------------*/

Timer::Timer(const std::string& name)
    : m_name(name), m_start(clock::now()), m_end(m_start), m_running(false)
{}

/*------------------------------------------------------------------------------------------*/

void Timer::start()
{
    m_start = clock::now();
    m_running = true;
}

/*------------------------------------------------------------------------------------------*/

void Timer::stop()
{
    if (m_running)
    {
        m_end = clock::now();
        m_running = false;
    }
}

/*------------------------------------------------------------------------------------------*/

void Timer::reset()
{
    m_start = clock::now();
    m_end = m_start;
    m_running = false;
}

/*------------------------------------------------------------------------------------------*/

double Timer::elapsedSeconds() const
{
    const auto end = m_running ? clock::now() : m_end;
    double seconds = std::chrono::duration<double>(end - m_start).count();
    std::cout << m_name << " elapsed time: " << seconds << " s\n";
    return seconds;
}

/*------------------------------------------------------------------------------------------*/

long long Timer::elapsedMilliseconds(bool print) const
{
    const auto end = m_running ? clock::now() : m_end;
    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - m_start).count();
    if (print)
    {
        std::cout << m_name << " elapsed time: " << ms << " ms\n";
    }
    return ms;
}

/*------------------------------------------------------------------------------------------*/

long long Timer::elapsedMicroseconds(bool print) const
{
    const auto end = m_running ? clock::now() : m_end;
    long long us = std::chrono::duration_cast<std::chrono::microseconds>(end - m_start).count();
    if (print)
    {
        std::cout << m_name << " elapsed time: " << us << " μs\n";
    }
    return us;
}
