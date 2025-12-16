#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>

#include "core/interfaces/NSInterfaces.hpp"
#include "core/lattice/NSLattice.hpp"
#include "core/CudaHelpers.hpp"

// Basic velocity dirichlet: overwrite to equilibrium on mask
template <typename Descriptor>
class VelocityDirichletEq : public NSBoundaryCondition<Descriptor> {
public:
    VelocityDirichletEq(const LBMParams& p,
                        std::vector<std::uint8_t> mask,
                        DATA_TYPE ux, DATA_TYPE uy, DATA_TYPE uz)
        : params(p), h_mask(std::move(mask)), u0x(ux), u0y(uy), u0z(uz)
    {
        const int n = p.Nx*p.Ny*p.Nz;
        if ((int)h_mask.size()!=n) throw std::runtime_error("VelocityDirichletEq mask mismatch");

        d_mask = static_cast<std::uint8_t*>(cudaMallocBytes(h_mask.size()*sizeof(std::uint8_t)));
        cudaMemcpyHtoD(d_mask, h_mask.data(), h_mask.size()*sizeof(std::uint8_t));
    }

    ~VelocityDirichletEq() override { cudaFreeBytes(d_mask); }

    void apply(Lattice<Descriptor>& lat, BoundaryPhase phase, int step) override;

private:
    LBMParams params{};
    std::vector<std::uint8_t> h_mask;
    std::uint8_t* d_mask = nullptr;

    DATA_TYPE u0x=DATA_TYPE(0), u0y=DATA_TYPE(0), u0z=DATA_TYPE(0);
};
