#pragma once

/* this file contains some Internal parameters and definitions */

//this is the Data type used, tested for float and double
using DATA_TYPE = double;


//These are some terminal colors and reset stuff used for fancy printing
namespace termcolor {
    constexpr const char* reset  = "\033[0m";
    constexpr const char* bold   = "\033[1m";

    constexpr const char* gray   = "\033[90m";
    constexpr const char* red    = "\033[31m";
    constexpr const char* green  = "\033[32m";
    constexpr const char* yellow = "\033[33m";
    constexpr const char* blue   = "\033[34m";
}