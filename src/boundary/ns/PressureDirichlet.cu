#include <vector>
#include <cstdint>

#include "boundary/ns/PressureDirichlet.hpp"
#include "core/descriptors/D3Q19.hpp"

#include "core/CudaHelpers.hpp"
#include "core/Types.hpp"
#include "core/Params.hpp"
#include "core/lattice/NSLattice.hpp"

#include <cuda_runtime.h>

extern __constant__ int c19_cx[19];
extern __constant__ int c19_cy[19];
extern __constant__ int c19_cz[19];
extern __constant__ DATA_TYPE c19_w[19];

// tiny 3D -> 1D helper
__device__ __forceinline__ int idx3(int x, int y, int z, int Nx, int Ny)
{
    return x + Nx * (y + Ny * z);
}

// pressure dirichlet at outlet (set rho, take u from neighbor)
__global__ void k_pressure_dirichlet_outlet_d3q19(
    DATA_TYPE* f,
    const std::uint8_t* mask,
    int Nx, int Ny, int Nz,
    DATA_TYPE rho_out
){
    // global cell index
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    int nCells = Nx * Ny * Nz;

    // out of range / not in mask
    if (cell >= nCells) return;
    if (!mask[cell]) return;

    // idx -> (x,y,z)
    int z = cell / (Nx * Ny);
    int rem = cell - z * (Nx * Ny);
    int y = rem / Nx;
    int x = rem - y * Nx;

    // need neighbor at x-1
    if (x <= 0) return;

    // neighbor cell (inside domain)
    int cellN = idx3(x - 1, y, z, Nx, Ny);

    // compute neighbor rho + u from f
    DATA_TYPE rhoN = DATA_TYPE(0);
    DATA_TYPE ux = DATA_TYPE(0);
    DATA_TYPE uy = DATA_TYPE(0);
    DATA_TYPE uz = DATA_TYPE(0);

    int baseN = cellN * 19;
    #pragma unroll
    for (int q = 0; q < 19; ++q) {
        DATA_TYPE fi = f[baseN + q];
        rhoN += fi;
        ux += fi * DATA_TYPE(c19_cx[q]);
        uy += fi * DATA_TYPE(c19_cy[q]);
        uz += fi * DATA_TYPE(c19_cz[q]);
    }

    // normalize u (avoid div0)
    if (rhoN > DATA_TYPE(0)) {
        DATA_TYPE inv = DATA_TYPE(1) / rhoN;
        ux *= inv; uy *= inv; uz *= inv;
    } else {
        ux = uy = uz = DATA_TYPE(0);
    }

    // write equilibrium with rho_out + neighbor u
    const DATA_TYPE u2 = ux * ux + uy * uy + uz * uz;
    int base = cell * 19;

    #pragma unroll
    for (int q = 0; q < 19; ++q)
    {
        DATA_TYPE eu =
            DATA_TYPE(c19_cx[q]) * ux +
            DATA_TYPE(c19_cy[q]) * uy +
            DATA_TYPE(c19_cz[q]) * uz;

        DATA_TYPE feq = c19_w[q] * rho_out *
                        (DATA_TYPE(1)
                        + DATA_TYPE(3)   * eu
                        + DATA_TYPE(4.5) * eu * eu
                        - DATA_TYPE(1.5) * u2);

        f[base + q] = feq;
    }
}


template<>
PressureDirichlet<D3Q19>::PressureDirichlet(
    const LBMParams& p,
    std::vector<std::uint8_t> mask,
    DATA_TYPE rho_out_
)
: params(p),
  mask_h(std::move(mask)),
  rho_out(rho_out_)
{
    // upload mask to device
    const std::size_t bytes = mask_h.size() * sizeof(std::uint8_t);
    mask_d = static_cast<std::uint8_t*>(cudaMallocBytes(bytes));
    cudaMemcpyHtoD(mask_d, mask_h.data(), bytes);
}

template<>
PressureDirichlet<D3Q19>::~PressureDirichlet()
{
    // cleanup mask
    cudaFreeBytes(mask_d);
    mask_d = nullptr;
}

template<>
void PressureDirichlet<D3Q19>::apply(Lattice<D3Q19>& lat, BoundaryPhase phase, int /*step*/)
{
    // only after streaming
    if (phase != BoundaryPhase::PostStreaming) return;

    // launch config
    const auto& p = lat.P();
    const int n = p.Nx * p.Ny * p.Nz;
    const int block = 256;
    const int grid  = (n + block - 1) / block;

    // apply outlet rho BC
    k_pressure_dirichlet_outlet_d3q19<<<grid, block>>>(
        lat.d_f_new_ptr(),
        mask_d,
        p.Nx, p.Ny, p.Nz,
        rho_out
    );

    // check kernel launch
    cudaCheckThrow("PressureDirichlet outlet D3Q19");
}

template class PressureDirichlet<D3Q19>;
