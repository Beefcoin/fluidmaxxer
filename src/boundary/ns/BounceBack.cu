#include "boundary/ns/BounceBack.hpp"
#include "core/descriptors/D3Q19.hpp"
#include "core/CudaHelpers.hpp"
#include "core/Types.hpp"
#include <cuda_runtime.h>
#include <cstdint>

extern __constant__ int c19_opp[19];

/* bounceback after streaming, this is not nice as currently it is only applicable for d3q19 
but i will change it soon, so it is based on the descriptor definition */
__global__ void k_bounceback_post_d3q19(
    DATA_TYPE* __restrict__ f_new,
    const std::uint8_t* __restrict__ mask,
    int nCells
){
    // global cell index
    int cell = blockIdx.x * blockDim.x + threadIdx.x;

    // out of range / not obstacle
    if (cell >= nCells) return;
    if (!mask[cell]) return;

    // base index for distributions
    int base = cell * 19;

    // reflect: q <- opp(q)
    #pragma unroll
    for (int q = 0; q < 19; ++q)
        f_new[base + q] = f_new[base + c19_opp[q]];
}

template<>
void BounceBack<D3Q19>::apply(Lattice<D3Q19>& lat, BoundaryPhase phase, int)
{
    // only after streaming
    if (phase != BoundaryPhase::PostStreaming) return;

    // launch config
    int n = lat.Size();
    int block = 256;
    int grid  = (n + block - 1) / block;

    // apply BB on device
    k_bounceback_post_d3q19<<<grid, block>>>(
        lat.d_f_new_ptr(),
        d_mask,
        n
    );

    // check kernel launch
    cudaCheckThrow("k_bounceback_post_d3q19");
}

template class BounceBack<D3Q19>;
