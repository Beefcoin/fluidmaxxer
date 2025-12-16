#include "reaction/LinearLDF.hpp"
#include "core/CudaHelpers.hpp"
#include "core/Types.hpp"
#include <cuda_runtime.h>
#include <cstdint>

// Henry (linear) + LDF (like Langmuir version):
// c = max(phi - phiMinClamp, 0)
// q_eq = K * c
// dq = kLDF * (q_eq - q)
// q += dq
// Sphi = -dq
__global__ void k_henry_ldf_source(
    int nCells,
    const std::uint8_t* __restrict__ region,
    const DATA_TYPE* __restrict__ phi,
    DATA_TYPE* __restrict__ Sphi,
    DATA_TYPE* __restrict__ q,
    DATA_TYPE K,
    DATA_TYPE kLDF,
    DATA_TYPE phiMinClamp
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;

    // outside region -> no source
    if (!region[i])
    {
        Sphi[i] = DATA_TYPE(0);
        return;
    }

    // clamp phi and shift to avoid negative stuff
    DATA_TYPE c = phi[i];
    if (c < phiMinClamp) c = phiMinClamp;
    DATA_TYPE cEff = c - phiMinClamp;      // >= 0

    // Henry eq loading
    DATA_TYPE qeq = K * cEff;

    // LDF update
    DATA_TYPE qi = q[i];
    DATA_TYPE dq = kLDF * (qeq - qi);
    q[i] = qi + dq;

    // sink in fluid
    Sphi[i] = -dq;
}

void launch_henry_ldf_source_d3q7(
    int nCells,
    const std::uint8_t* d_regionMask,
    const DATA_TYPE* d_phi,
    DATA_TYPE* d_Sphi,
    DATA_TYPE* d_q,
    DATA_TYPE K,
    DATA_TYPE kLDF,
    DATA_TYPE phiMinClamp
){
    int block = 256;
    int grid  = (nCells + block - 1) / block;

    k_henry_ldf_source<<<grid, block>>>(
        nCells,
        d_regionMask,
        d_phi,
        d_Sphi,
        d_q,
        K,
        kLDF,
        phiMinClamp
    );

    cudaCheckThrow("k_henry_ldf_source");
}
