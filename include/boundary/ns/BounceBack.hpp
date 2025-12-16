#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>

#include "core/interfaces/NSInterfaces.hpp"
#include "core/lattice/NSLattice.hpp"
#include "core/CudaHelpers.hpp"

// BounceBack on mask (PostStreaming)
template <typename Descriptor>
class BounceBack : public NSBoundaryCondition<Descriptor> {
public:
    BounceBack(const LBMParams& p, std::vector<std::uint8_t> mask)
        : params(p), h_mask(std::move(mask))
    {
        // mask size sanity
        const int n = p.Nx * p.Ny * p.Nz;
        if ((int)h_mask.size() != n)
            throw std::runtime_error("BounceBack mask mismatch");

        // upload mask to device
        const std::size_t bytes = h_mask.size() * sizeof(std::uint8_t);
        d_mask = static_cast<std::uint8_t*>(cudaMallocBytes(bytes));
        cudaMemcpyHtoD(d_mask, h_mask.data(), bytes);
    }

    ~BounceBack() override
    {
        // cleanup
        cudaFreeBytes(d_mask);
    }

    // apply BC (kernel in .cu)
    void apply(Lattice<Descriptor>& lat, BoundaryPhase phase, int step) override;

private:
    // params copy
    LBMParams params{};

    // host,device mask
    std::vector<std::uint8_t> h_mask;
    std::uint8_t* d_mask = nullptr;
};
