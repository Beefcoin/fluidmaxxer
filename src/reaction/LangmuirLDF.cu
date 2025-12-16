#include "reaction/LangmuirLDF.hpp"
#include "core/CudaHelpers.hpp"
#include "core/Types.hpp"
#include <cuda_runtime.h>
#include <cstdint>

__global__ void k_langmuir_ldf_source(
    int nCells,
    const std::uint8_t* __restrict__ region,
    const DATA_TYPE* __restrict__ phi,
    DATA_TYPE* __restrict__ Sphi,
    DATA_TYPE* __restrict__ q,
    DATA_TYPE qMax, DATA_TYPE b, DATA_TYPE kLDF,
    DATA_TYPE phiMinClamp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;

    if (!region[i])
    {
        Sphi[i] = DATA_TYPE(0);
        return;
    }

    DATA_TYPE c = phi[i];
    if (c < phiMinClamp) c = phiMinClamp;
    DATA_TYPE clampedCorrected = c-phiMinClamp;

    // Langmuir equilibrium q* = qMax * b c / (1 + b c)
    DATA_TYPE bc = b * clampedCorrected;
    DATA_TYPE qeq = qMax * (bc / (DATA_TYPE(1) + bc));

    DATA_TYPE qi = q[i];

    // LDF: dq/dt = k (q* - q)
    DATA_TYPE dq = kLDF * (qeq - qi);
    q[i] = qi + dq;

    // Mass conservation coupling: dphi/dt gets -dq (sink in fluid concentration)
    Sphi[i] = -dq;
}

void launch_langmuir_ldf_source_d3q7(
    int nCells,
    const std::uint8_t* d_regionMask,
    const DATA_TYPE* d_phi,
    DATA_TYPE* d_Sphi,
    DATA_TYPE* d_q,
    DATA_TYPE qMax, DATA_TYPE b, DATA_TYPE kLDF,
    DATA_TYPE phiMinClamp)
{
    int block = 256;
    int grid  = (nCells + block - 1) / block;
    k_langmuir_ldf_source<<<grid, block>>>(
        nCells, d_regionMask, d_phi, d_Sphi, d_q,
        qMax, b, kLDF, phiMinClamp
    );
    cudaCheckThrow("k_langmuir_ldf_source");
}
