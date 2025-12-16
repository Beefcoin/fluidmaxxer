#pragma once
#include <iostream>

// color helper fancy shit
namespace fancy
{
    // basic styles
    constexpr const char *reset = "\033[0m";
    constexpr const char *bold = "\033[1m";

    // colors
    constexpr const char *red = "\033[31m";
    constexpr const char *green = "\033[32m";
    constexpr const char *yellow = "\033[33m";
    constexpr const char *blue = "\033[34m";

    inline std::ostream &unitConverterTag(std::ostream &os = std::cout)
    {
        return os << bold << blue << "[UnitConverter]: " << reset;
    }
    inline std::ostream &mainTag(std::ostream &os = std::cout)
    {
        return os << bold << red << "[Main]: " << reset;
    }
    inline std::ostream &VTIWriterTag(std::ostream &os = std::cout)
    {
        return os << bold << green << "[VTIWRITER]: " << reset;
    }
    inline std::ostream &timerTag(std::ostream &os = std::cout)
    {
        return os << bold << blue << "[Timer]: " << reset;
    }

    // ON / OFF toggle (bool)
    inline std::ostream &onOff(std::ostream &os, bool on)
    {
        if (on)
            return os << bold << green << "ON" << reset;
        else
            return os << bold << red << "OFF" << reset;
    }


    inline std::ostream &infoTag(std::ostream &os = std::cout)
    {
        return os << bold << blue << "[info]" << reset;
    }

    inline std::ostream &warnTag(std::ostream &os = std::cout)
    {
        return os << bold << yellow << "[warn]" << reset;
    }

    inline std::ostream &errTag(std::ostream &os = std::cout)
    {
        return os << bold << red << "[error]" << reset;
    }
}
