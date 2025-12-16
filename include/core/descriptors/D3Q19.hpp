#pragma once
#include <array>
#include "core/Types.hpp"

struct D3Q19 {
    //discrete velocity directions
    static constexpr int Q = 19;

    //speed of sound squared and inverse
    static constexpr DATA_TYPE cs2     = DATA_TYPE(1) / DATA_TYPE(3);
    static constexpr DATA_TYPE inv_cs2 = DATA_TYPE(3);

    // Discrete lattice velocity components c_q = (cx[q], cy[q], cz[q]) in lattice units
    static constexpr std::array<int, Q> cx = {
        0,  1,-1, 0, 0, 0, 0,   1,-1, 1,-1,   1,-1, 1,-1,  0, 0, 0, 0
    };
    static constexpr std::array<int, Q> cy = {
        0,  0, 0, 1,-1, 0, 0,   1, 1,-1,-1,   0, 0, 0, 0,  1,-1, 1,-1
    };
    static constexpr std::array<int, Q> cz = {
        0,  0, 0, 0, 0, 1,-1,   0, 0, 0, 0,   1, 1,-1,-1,  1, 1,-1,-1
    };

    //Descriptor weights
    static constexpr std::array<DATA_TYPE, Q> w = {
        DATA_TYPE(1)/DATA_TYPE(3),

        DATA_TYPE(1)/DATA_TYPE(18), DATA_TYPE(1)/DATA_TYPE(18),
        DATA_TYPE(1)/DATA_TYPE(18), DATA_TYPE(1)/DATA_TYPE(18),
        DATA_TYPE(1)/DATA_TYPE(18), DATA_TYPE(1)/DATA_TYPE(18),

        DATA_TYPE(1)/DATA_TYPE(36), DATA_TYPE(1)/DATA_TYPE(36),
        DATA_TYPE(1)/DATA_TYPE(36), DATA_TYPE(1)/DATA_TYPE(36),

        DATA_TYPE(1)/DATA_TYPE(36), DATA_TYPE(1)/DATA_TYPE(36),
        DATA_TYPE(1)/DATA_TYPE(36), DATA_TYPE(1)/DATA_TYPE(36),

        DATA_TYPE(1)/DATA_TYPE(36), DATA_TYPE(1)/DATA_TYPE(36),
        DATA_TYPE(1)/DATA_TYPE(36), DATA_TYPE(1)/DATA_TYPE(36)
    };

    //opposite directions for BounceBack and stuff like that
    static constexpr std::array<int, Q> opp = {
        0,
        2,1,4,3,6,5,
        10,9,8,7,
        14,13,12,11,
        18,17,16,15
    };
};

void uploadD3Q19Constants();
