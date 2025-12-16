#include "boundary/ns/VelocityDirichletNEQ.hpp"
#include "core/descriptors/D3Q19.hpp"
#include "core/CudaHelpers.hpp"
#include "core/Types.hpp"

#include <cuda_runtime.h>

extern __constant__ int       c19_cx[19];
extern __constant__ int       c19_cy[19];
extern __constant__ int       c19_cz[19];
extern __constant__ DATA_TYPE c19_w[19];

__device__ __forceinline__ void compute_rho_u_d3q19(
    const DATA_TYPE* __restrict__ f, int base,
    DATA_TYPE& rho, DATA_TYPE& ux, DATA_TYPE& uy, DATA_TYPE& uz
){
    DATA_TYPE r = DATA_TYPE(0);
    DATA_TYPE jx = DATA_TYPE(0), jy = DATA_TYPE(0), jz = DATA_TYPE(0);

    #pragma unroll
    for (int q = 0; q < 19; ++q) {
        DATA_TYPE fq = f[base + q];
        r  += fq;
        jx += fq * DATA_TYPE(c19_cx[q]);
        jy += fq * DATA_TYPE(c19_cy[q]);
        jz += fq * DATA_TYPE(c19_cz[q]);
    }
    rho = r;
    const DATA_TYPE inv = (r > DATA_TYPE(0)) ? (DATA_TYPE(1) / r) : DATA_TYPE(0);
    ux = jx * inv;
    uy = jy * inv;
    uz = jz * inv;
}

__device__ __forceinline__ DATA_TYPE feq_d3q19(
    int q, DATA_TYPE rho, DATA_TYPE ux, DATA_TYPE uy, DATA_TYPE uz
){
    const DATA_TYPE u2 = ux*ux + uy*uy + uz*uz;
    const DATA_TYPE eu =
        DATA_TYPE(c19_cx[q]) * ux +
        DATA_TYPE(c19_cy[q]) * uy +
        DATA_TYPE(c19_cz[q]) * uz;
    const DATA_TYPE cu = DATA_TYPE(3) * eu;
    return c19_w[q] * rho *
           (DATA_TYPE(1) + cu + DATA_TYPE(0.5)*cu*cu - DATA_TYPE(1.5)*u2);
}

// velocity dirichlet (NEQ extrapolation) before collision
__global__ void k_velocity_dirichlet_neq_pre_d3q19(
    DATA_TYPE* __restrict__ f,
    const int* __restrict__ bndIdx,
    const int* __restrict__ neiIdx,
    int nBnd,
    DATA_TYPE ux, DATA_TYPE uy, DATA_TYPE uz
){
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nBnd) return;

    // boundary cell + its "inner" neighbor cell (one node into fluid)
    const int cell = bndIdx[k];
    const int nei  = neiIdx[k];

    const int baseB = cell * 19;
    const int baseN = nei  * 19;

    // compute neighbor macros from its populations:
    // rhoN = sum_q fN[q], uN = (sum_q fN[q]*c_q) / rhoN
    DATA_TYPE rhoN, uxN, uyN, uzN;
    compute_rho_u_d3q19(f, baseN, rhoN, uxN, uyN, uzN);

    // NEQ extrapolation:
    // split neighbor distributions into equilibrium + non-equilibrium part:
    //   fN[q] = feq(rhoN, uN)[q] + fneqN[q]
    // => fneqN[q] = fN[q] - feq(rhoN, uN)[q]
    //
    // then write boundary distributions as:
    //   fB[q] = feq(rhoN, uBC)[q] + fneqN[q]
    //
    // enforce Dirichlet velocity through feq(rhoN, uBC),
    // but keep shear/stress info from neighbor via fneqN.
    #pragma unroll
    for (int q = 0; q < 19; ++q) {
        const DATA_TYPE fN   = f[baseN + q];

        // equilibrium using neighbor macros (rhoN, uN)
        const DATA_TYPE feqN = feq_d3q19(q, rhoN, uxN, uyN, uzN);

        // equilibrium using imposed boundary velocity (rhoN, uBC)
        // NOTE: we reuse rhoN here (common choice for velocity BCs)
        const DATA_TYPE feqB = feq_d3q19(q, rhoN, ux,  uy,  uz);

        // copy neighbor non-equilibrium part to boundary:
        // fB = feqB + (fN - feqN)
        f[baseB + q] = feqB + (fN - feqN);
    }
}

template<>
void VelocityDirichletNEQ<D3Q19>::apply(Lattice<D3Q19>& lat, BoundaryPhase phase, int)
{
    if (phase != BoundaryPhase::PreCollision) return;
    if (nBnd == 0) return;

    int block = 256;
    int grid  = (nBnd + block - 1) / block;

    k_velocity_dirichlet_neq_pre_d3q19<<<grid, block>>>(
        lat.d_f_ptr(),
        d_bndIdx,
        d_neiIdx,
        nBnd,
        u0x, u0y, u0z
    );
    cudaCheckThrow("k_velocity_dirichlet_neq_pre_d3q19");
}

template class VelocityDirichletNEQ<D3Q19>;
