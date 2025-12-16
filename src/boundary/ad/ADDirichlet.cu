#include "boundary/ad/ADDirichlet.hpp"
#include "core/descriptors/D3Q7.hpp"
#include "core/CudaHelpers.hpp"
#include "core/Types.hpp"
#include <cuda_runtime.h>
#include <cstdint>

extern __constant__ DATA_TYPE c7_w[7];

// Dirichlet BC kernel 
__global__ void k_ad_dirichlet(
    DATA_TYPE* g_new,
    const std::uint8_t* mask,
    int n,
    DATA_TYPE phi
){
    // global cell index
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    // out of range or not in BC mask
    if (c >= n || !mask[c]) return;

    // base index for distributions
    int base = c * 7;

    // write weighted distributions so sum_q g = phi
    #pragma unroll
    for (int q = 0; q < 7; ++q)
        g_new[base + q] = phi * c7_w[q];
}

template<>
void ADDirichlet<D3Q7>::apply(ADLattice<D3Q7>& lat, BoundaryPhase phase, int)
{
    // only apply after streaming
    if (phase != BoundaryPhase::PostStreaming) return;

    // launch config (might experiment a little bit with this soon)
    int n = lat.Size();
    int block = 256;
    int grid  = (n + block - 1) / block;

    // apply BC on device
    k_ad_dirichlet<<<grid, block>>>(
        lat.d_g_new_ptr(),
        d_mask,
        n,
        phi0
    );

    // check kernel launch
    cudaCheckThrow("k_ad_dirichlet");
}

template class ADDirichlet<D3Q7>;
