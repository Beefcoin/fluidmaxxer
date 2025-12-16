#pragma once
#include "core/Params.hpp"
#include "reaction/AdsorbedField.hpp"
#include "core/lattice/ADLattice.hpp"
#include <cstdint>
// GPU kernel launcher (implemented in .cu)
void launch_langmuir_ldf_source_d3q7(
    int nCells,
    const std::uint8_t* d_regionMask,     // adsorption region 
    const DATA_TYPE* d_phi,               // AD scalar field
    DATA_TYPE* d_Sphi,                    // source term for phi
    DATA_TYPE* d_q,                       // adsorbed amount
    DATA_TYPE qMax, DATA_TYPE b, DATA_TYPE kLDF,
    DATA_TYPE phiMinClamp);

template<typename ADDesc>
class LangmuirLDF
{
public:
    explicit LangmuirLDF(const LBMParams& p) : p_(p) {}

    void computeSource(ADLattice<ADDesc>& ad, AdsorbedField& q, const std::uint8_t* d_regionMask)
    {
        const int n = p_.Nx * p_.Ny * p_.Nz;
        launch_langmuir_ldf_source_d3q7(
            n,
            d_regionMask,
            ad.d_phi_ptr(),
            ad.d_Sphi_ptr(),
            q.d_q(),
            p_.qMax, p_.bLangmuir, p_.kLDF,
            p_.minConcentration
        );
    }

private:
    LBMParams p_;
};
