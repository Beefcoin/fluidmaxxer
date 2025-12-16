#pragma once
#include "core/Params.hpp"
#include "reaction/AdsorbedField.hpp"
#include "core/lattice/ADLattice.hpp"
#include <cstdint>

// GPU kernel launcher (defined in .cu)
void launch_henry_ldf_source_d3q7(
    int nCells,
    const std::uint8_t* d_regionMask, // adsorption region
    const DATA_TYPE* d_phi,           // AD scalar field
    DATA_TYPE* d_Sphi,                // source term for phi
    DATA_TYPE* d_q,                   // adsorbed amount (updated in kernel)
    DATA_TYPE K,                      // Henry constant
    DATA_TYPE kLDF,                   // LDF rate
    DATA_TYPE phiMinClamp             // small clamp
);

template<typename ADDesc>
class LinearLDF
{
public:
    explicit LinearLDF(const LBMParams& p)
        : p_(p) {}

    // compute adsorption / desorption (Henry + LDF)
    void computeSource(
        ADLattice<ADDesc>& ad,
        AdsorbedField& q,
        const std::uint8_t* d_regionMask
    )
    {
        const int n = p_.Nx * p_.Ny * p_.Nz;

        launch_henry_ldf_source_d3q7(
            n,
            d_regionMask,
            ad.d_phi_ptr(),     // concentration
            ad.d_Sphi_ptr(),    // phi source
            q.d_q(),            // q field 
            p_.KHenry,
            p_.kA,             
            p_.minConcentration
        );
    }

private:
    // params copy
    LBMParams p_;
};
