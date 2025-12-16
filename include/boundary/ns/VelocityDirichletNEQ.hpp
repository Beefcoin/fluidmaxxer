#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>

#include "core/interfaces/NSInterfaces.hpp"
#include "core/lattice/NSLattice.hpp"
#include "core/CudaHelpers.hpp"
#include "core/Types.hpp"

// OpenLB-like LocalVelocity via NEQ extrapolation.
// Drop-in signature: (p, mask, ux, uy, uz)
// CUDA kernels + apply specialization are in .cu
template <typename Descriptor>
class VelocityDirichletNEQ : public NSBoundaryCondition<Descriptor> {
public:
    VelocityDirichletNEQ(const LBMParams& p,
                         std::vector<std::uint8_t> mask,
                         DATA_TYPE ux, DATA_TYPE uy, DATA_TYPE uz)
        : params(p), h_mask(std::move(mask)), u0x(ux), u0y(uy), u0z(uz)
    {
        const int n = params.Nx * params.Ny * params.Nz;
        if ((int)h_mask.size() != n)
            throw std::runtime_error("VelocityDirichletNEQ mask mismatch");

        // compress boundary indices
        h_bndIdx.reserve(n / 10);
        for (int i = 0; i < n; ++i) if (h_mask[i]) h_bndIdx.push_back(i);
        nBnd = (int)h_bndIdx.size();
        if (nBnd == 0) return;

        // infer plane direction from mask
        inferPlaneAndBuildNeighbors();

        // upload lists
        d_bndIdx = static_cast<int*>(cudaMallocBytes(nBnd * sizeof(int)));
        d_neiIdx = static_cast<int*>(cudaMallocBytes(nBnd * sizeof(int)));
        cudaMemcpyHtoD(d_bndIdx, h_bndIdx.data(), nBnd * sizeof(int));
        cudaMemcpyHtoD(d_neiIdx, h_neiIdx.data(), nBnd * sizeof(int));
    }

    ~VelocityDirichletNEQ() override {
        cudaFreeBytes(d_bndIdx);
        cudaFreeBytes(d_neiIdx);
    }

    void apply(Lattice<Descriptor>& lat, BoundaryPhase phase, int step) override;

private:
    enum class PlaneDir : int { XMin, XMax, YMin, YMax, ZMin, ZMax };

    void inferPlaneAndBuildNeighbors()
    {
        const int nx = params.Nx, ny = params.Ny, nz = params.Nz;

        int minx = nx-1, maxx = 0;
        int miny = ny-1, maxy = 0;
        int minz = nz-1, maxz = 0;

        for (int cell : h_bndIdx) {
            const int x = cell % nx;
            const int y = (cell / nx) % ny;
            const int z = cell / (nx * ny);
            minx = std::min(minx, x); maxx = std::max(maxx, x);
            miny = std::min(miny, y); maxy = std::max(maxy, y);
            minz = std::min(minz, z); maxz = std::max(maxz, z);
        }

        bool isXPlane = (minx == maxx);
        bool isYPlane = (miny == maxy);
        bool isZPlane = (minz == maxz);

        PlaneDir dir;
        int planes = int(isXPlane) + int(isYPlane) + int(isZPlane);
        if (planes != 1) {
            throw std::runtime_error(
                "VelocityDirichletNEQ: mask must describe exactly one axis-aligned plane (x=const OR y=const OR z=const).");
        }

        if (isXPlane) dir = (minx == 0) ? PlaneDir::XMin : ((minx == nx-1) ? PlaneDir::XMax : throw std::runtime_error("VelocityDirichletNEQ: x-plane not at boundary"));
        if (isYPlane) dir = (miny == 0) ? PlaneDir::YMin : ((miny == ny-1) ? PlaneDir::YMax : throw std::runtime_error("VelocityDirichletNEQ: y-plane not at boundary"));
        if (isZPlane) dir = (minz == 0) ? PlaneDir::ZMin : ((minz == nz-1) ? PlaneDir::ZMax : throw std::runtime_error("VelocityDirichletNEQ: z-plane not at boundary"));

        h_neiIdx.resize(nBnd);

        for (int k = 0; k < nBnd; ++k) {
            const int cell = h_bndIdx[k];
            const int x = cell % nx;
            const int y = (cell / nx) % ny;
            const int z = cell / (nx * ny);

            int xn = x, yn = y, zn = z;
            switch (dir) {
                case PlaneDir::XMin: xn = x + 1; break;
                case PlaneDir::XMax: xn = x - 1; break;
                case PlaneDir::YMin: yn = y + 1; break;
                case PlaneDir::YMax: yn = y - 1; break;
                case PlaneDir::ZMin: zn = z + 1; break;
                case PlaneDir::ZMax: zn = z - 1; break;
            }

            if (xn < 0 || xn >= nx || yn < 0 || yn >= ny || zn < 0 || zn >= nz)
                throw std::runtime_error("VelocityDirichletNEQ: inferred neighbor out of domain.");

            h_neiIdx[k] = xn + yn * nx + zn * nx * ny;
        }
    }

private:
    LBMParams params{};
    std::vector<std::uint8_t> h_mask;

    std::vector<int> h_bndIdx;
    std::vector<int> h_neiIdx;
    int nBnd = 0;

    int* d_bndIdx = nullptr;
    int* d_neiIdx = nullptr;

    DATA_TYPE u0x=DATA_TYPE(0), u0y=DATA_TYPE(0), u0z=DATA_TYPE(0);
};
