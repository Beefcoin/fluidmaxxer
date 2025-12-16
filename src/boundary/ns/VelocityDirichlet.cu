//needsoverhaul
#include "boundary/ns/VelocityDirichlet.hpp"
#include "core/descriptors/D3Q19.hpp"
#include "core/CudaHelpers.hpp"
#include "core/Types.hpp"
#include <cuda_runtime.h>
#include <cstdint>

extern __constant__ int       c19_cx[19];
extern __constant__ int       c19_cy[19];
extern __constant__ int       c19_cz[19];
extern __constant__ DATA_TYPE c19_w[19];

// velocity dirichlet (eq fill) after streaming
__global__ void k_velocity_dirichlet_eq_post_d3q19(
    DATA_TYPE* __restrict__ f_new,
    const std::uint8_t* __restrict__ mask,
    int nCells,
    DATA_TYPE rho0,
    DATA_TYPE ux, DATA_TYPE uy, DATA_TYPE uz
){
    // global cell index
    int cell = blockIdx.x * blockDim.x + threadIdx.x;

    // out of range / not in mask
    if (cell >= nCells) return;
    if (!mask[cell]) return;

    // base index for distributions
    int base = cell * 19;

    // const u^2
    DATA_TYPE u2 = ux * ux + uy * uy + uz * uz;

    // write eq with given (rho0, u0)
    #pragma unroll
    for (int q = 0; q < 19; ++q) {
        DATA_TYPE eu =
            DATA_TYPE(c19_cx[q]) * ux +
            DATA_TYPE(c19_cy[q]) * uy +
            DATA_TYPE(c19_cz[q]) * uz;

        DATA_TYPE cu  = DATA_TYPE(3) * eu;
        DATA_TYPE feq = c19_w[q] * rho0 *
                        (DATA_TYPE(1)
                         + cu
                         + DATA_TYPE(0.5) * cu * cu
                         - DATA_TYPE(1.5) * u2);

        f_new[base + q] = feq;
    }
}

__global__ void k_velocity_dirichlet_eq_pre_d3q19(
    DATA_TYPE* __restrict__ f,
    const std::uint8_t* __restrict__ mask,
    int nCells,
    DATA_TYPE rho0,
    DATA_TYPE ux, DATA_TYPE uy, DATA_TYPE uz
){
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= nCells) return;
    if (!mask[cell]) return;

    int base = cell * 19;
    DATA_TYPE u2 = ux*ux + uy*uy + uz*uz;

    #pragma unroll
    for (int q = 0; q < 19; ++q) {
        DATA_TYPE eu =
            DATA_TYPE(c19_cx[q]) * ux +
            DATA_TYPE(c19_cy[q]) * uy +
            DATA_TYPE(c19_cz[q]) * uz;

        DATA_TYPE cu  = DATA_TYPE(3) * eu;
        DATA_TYPE feq = c19_w[q] * rho0 *
                        (DATA_TYPE(1)
                         + cu
                         + DATA_TYPE(0.5) * cu * cu
                         - DATA_TYPE(1.5) * u2);

        f[base + q] = feq;
    }
}


/* template<>
void VelocityDirichletEq<D3Q19>::apply(Lattice<D3Q19>& lat, BoundaryPhase phase, int)
{
    // only after streaming
    if (phase != BoundaryPhase::PostStreaming) return;

    // launch config
    int n = lat.Size();
    int block = 256;
    int grid  = (n + block - 1) / block;

    // apply BC on device
    k_velocity_dirichlet_eq_post_d3q19<<<grid, block>>>(
        lat.d_f_new_ptr(),
        d_mask,
        n,
        params.rho0,  
        u0x, u0y, u0z
    );

    // check kernel launch
    cudaCheckThrow("k_velocity_dirichlet_eq_post_d3q19");
} */

template<>
void VelocityDirichletEq<D3Q19>::apply(Lattice<D3Q19>& lat, BoundaryPhase phase, int)
{
    // only before collision (important for pull streaming)
    if (phase != BoundaryPhase::PreCollision) return;

    int n = lat.Size();
    int block = 256;
    int grid  = (n + block - 1) / block;

    k_velocity_dirichlet_eq_pre_d3q19<<<grid, block>>>(
        lat.d_f_ptr(),   // <-- IMPORTANT: write to f, not f_new
        d_mask,
        n,
        params.rho0,
        u0x, u0y, u0z
    );

    cudaCheckThrow("k_velocity_dirichlet_eq_pre_d3q19");
}


template class VelocityDirichletEq<D3Q19>;
