#pragma once
#include <string>
#include <vector>
#include <cstdint>

// x-plane mask (x = xPlane)
inline std::vector<std::uint8_t> buildXPlaneMask(int Nx, int Ny, int Nz, int xPlane)
{
    std::vector<std::uint8_t> m(Nx * Ny * Nz, 0);
    for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
            m[xPlane + Nx * (y + Ny * z)] = 1;
    return m;
}

// quick flag check
inline bool hasFlag(int argc, char** argv, const std::string& flag)
{
    for (int i = 1; i < argc; ++i)
        if (flag == argv[i]) return true;
    return false;
}
