#pragma once
#include <array>
#include "core/Types.hpp"

struct D3Q7 {
    //discrete velocity directions
    static constexpr int Q = 7;

    //speed of sound squared and inverse
    static constexpr DATA_TYPE cs2     = DATA_TYPE(1) / DATA_TYPE(4);
    static constexpr DATA_TYPE inv_cs2 = DATA_TYPE(4);

    // Discrete lattice velocity components c_q = (cx[q], cy[q], cz[q]) in lattice units
    static constexpr std::array<int,Q> cx = {0, 1,-1, 0, 0, 0, 0};
    static constexpr std::array<int,Q> cy = {0, 0, 0, 1,-1, 0, 0};
    static constexpr std::array<int,Q> cz = {0, 0, 0, 0, 0, 1,-1};

    //Descriptor weights
    static constexpr std::array<DATA_TYPE,Q> w = {
        DATA_TYPE(1)/DATA_TYPE(4),
        DATA_TYPE(1)/DATA_TYPE(8), DATA_TYPE(1)/DATA_TYPE(8),
        DATA_TYPE(1)/DATA_TYPE(8), DATA_TYPE(1)/DATA_TYPE(8),
        DATA_TYPE(1)/DATA_TYPE(8), DATA_TYPE(1)/DATA_TYPE(8)
    };

    //opposite directions for BounceBack and stuff like that
    static constexpr std::array<int,Q> opp = {0,2,1,4,3,6,5};
};

void uploadD3Q7Constants();
