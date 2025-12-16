#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>

#include "core/Params.hpp"
#include "core/CudaHelpers.hpp"
#include "core/interfaces/ADInterfaces.hpp"
#include "core/lattice/ADLattice.hpp"

/* this is a neumann boundary condition for scalar AD Lattice, it can be used for outflow profiles */

template <typename Descriptor>
class ADNeumannOutlet : public ADBoundaryCondition<Descriptor> {
public:
    ADNeumannOutlet(const LBMParams& p,
                    std::vector<std::uint8_t> mask)
        : params(p), h_mask(std::move(mask))
    {
        const int n = p.Nx*p.Ny*p.Nz;
        //check if mask is valid
        if ((int)h_mask.size()!=n) throw std::runtime_error("ADNeumannOutlet mask mismatch");

        //allocate memory for the BC mask
        d_mask = static_cast<std::uint8_t*>(cudaMallocBytes(h_mask.size()*sizeof(std::uint8_t)));
        //copy mask data to device
        cudaMemcpyHtoD(d_mask, h_mask.data(), h_mask.size()*sizeof(std::uint8_t));
    }

    ~ADNeumannOutlet() override { cudaFreeBytes(d_mask); }

    void apply(ADLattice<Descriptor>& lat, BoundaryPhase phase, int step) override;

private:
    LBMParams params{};
    std::vector<std::uint8_t> h_mask;
    std::uint8_t* d_mask = nullptr;
};
