//needsoverhaul

// src/unitconverter/ad_unitconverter.h
#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include "core/Types.hpp"

// Unit converter for advection-diffusion LBM
// Computes tau_ad from physical diffusivity using Descriptor::cs2
// Can build a per-cell tau field using a mask for different diffusivities

template <typename Descriptor>
struct ADUnitConverter
{
    // grid
    int Nx, Ny, Nz;

    // from fluid unit conv.
    double dx_phys;
    double dt_phys;

    // lattice cs^2 from descriptor
    static constexpr double cs2_lbm = static_cast<double>(Descriptor::cs2);

    ADUnitConverter(int Nx_, int Ny_, int Nz_,
                    double dx_phys_,
                    double dt_phys_)
        : Nx(Nx_), Ny(Ny_), Nz(Nz_), dx_phys(dx_phys_), dt_phys(dt_phys_)
    {
        if (Nx < 2)
            throw std::runtime_error("ADUnitConverter: Nx must be >= 2");
        if (dx_phys <= 0.0)
            throw std::runtime_error("ADUnitConverter: dx_phys must be > 0");
        if (dt_phys <= 0.0)
            throw std::runtime_error("ADUnitConverter: dt_phys must be > 0");
        if (cs2_lbm <= 0.0)
            throw std::runtime_error("ADUnitConverter: Descriptor::cs2 must be > 0");
    }

    // D_phys [m^2/s] -> D_lbm
    double diffusivityPhysToLattice(double D_phys) const
    {
        return D_phys * dt_phys / (dx_phys * dx_phys);
    }

    // tau from D_phys
    double tauFromDiffusivityPhys(double D_phys) const
    {
        const double D_lbm = diffusivityPhysToLattice(D_phys);
        const double tau = D_lbm / cs2_lbm + 0.5;
        if (tau <= 0.5)
            throw std::runtime_error("ADUnitConverter: tau_ad <= 0.5 (unstable)");
        return tau;
    }

    // build tau field from a mask
    // mask == 1 -> solid, mask == 0 -> fluid
    // used for different Diffusivities
    std::vector<DATA_TYPE> buildTauField(const std::vector<std::uint8_t> &mask,
                                         double D_fluid_phys,
                                         double D_solid_phys) const
    {
        const std::size_t size = static_cast<std::size_t>(Nx) * Ny * Nz;
        if (mask.size() != size)
            throw std::runtime_error("ADUnitConverter: mask size mismatch");

        const DATA_TYPE tauFluid = static_cast<DATA_TYPE>(tauFromDiffusivityPhys(D_fluid_phys));
        const DATA_TYPE tauSolid = static_cast<DATA_TYPE>(tauFromDiffusivityPhys(D_solid_phys));

        using namespace termcolor;
        std::cout << "\n--------------------------------------------------\n";
        std::cout << green << bold <<" ADUnitConverter :: buildTauField\n"
                  << reset;
        std::cout << "--------------------------------------------------\n";

        std::cout << std::scientific << std::setprecision(5);


        std::cout << termcolor::yellow << termcolor::bold <<"-- Fluid\n"
                  << termcolor::reset;
        std::cout << " D_phys      : " << D_fluid_phys << "   m^2/s\n";
        std::cout << " tau         : " << tauFluid << "\n";

        std::cout << termcolor::yellow << termcolor::bold <<"-- Solid\n"
                  << termcolor::reset;
        std::cout << " D_phys      : " << D_solid_phys << "   m^2/s\n";
        std::cout << " tau         : " << tauSolid << "\n";

        std::cout << termcolor::yellow << termcolor::bold <<"-- Descriptor\n"
                  << termcolor::reset;
        std::cout << " cs^2        : " << cs2_lbm << "\n";

        std::cout << "--------------------------------------------------\n\n";

        std::vector<DATA_TYPE> tauField(size);
        for (std::size_t i = 0; i < size; ++i)
            tauField[i] = mask[i] ? tauSolid : tauFluid;

        return tauField;
    }
};
