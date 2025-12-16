#pragma once
#include "core/Types.hpp"

/* this file contains the LBMParams struct, it is used for a lot of stuff like configuring BCs and Collision Operators */

struct LBMParams
{
    // domain Size
    int Nx, Ny, Nz;

    // NS-Lattice configuration (density, relaxation time)
    DATA_TYPE rho0;
    DATA_TYPE tau;

    // AD Settings
    DATA_TYPE tau_ad;
    DATA_TYPE phi0;
    DATA_TYPE minConcentration = 1e-8;

    // periodicy flags for streaming
    bool periodicX = true;
    bool periodicY = true;
    bool periodicZ = true;

    // ADsorption constants
    bool enableAdsorption = false;
    DATA_TYPE KHenry; // Henry Constant
    DATA_TYPE kA;     // LDF rate (per step)

    DATA_TYPE qMax;      // Langmuir capacity
    DATA_TYPE bLangmuir; // Langmuir affinity
    DATA_TYPE kLDF;      // LDF rate (per step)

    // Porosity (Implementation sucks for now)
    DATA_TYPE porosityValue = 0.67;
    DATA_TYPE betaPorous = 1;
};

// These are BoundaryPhases that can be used to make a boundary condittion happen at a specific time
enum class BoundaryPhase
{
    PreCollision,
    PostStreaming
};
