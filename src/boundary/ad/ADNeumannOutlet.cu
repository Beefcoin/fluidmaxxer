#include "boundary/ad/ADNeumannOutlet.hpp"
#include "core/descriptors/D3Q7.hpp"

// Neumann type outlet, we copy pops from inner neighbor, this is just for outlets
__global__ void k_ad_neumann(
    DATA_TYPE* g_new,
    int Nx, int Ny, int Nz,
    const std::uint8_t* mask
){
    // global cell index
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    // only BC cells
    if (!mask[c]) return;

    // idx -> (x,y,z)
    int z = c / (Nx * Ny);
    int y = (c - z * Nx * Ny) / Nx;
    int x = c % Nx;

    // only on x+ boundary
    if (x != Nx - 1) return;

    // copy from x = Nx-2 (zero gradient)
    int src = (x - 1) + Nx * (y + Ny * z);

    // copy all directions
    #pragma unroll
    for (int q = 0; q < 7; ++q)
        g_new[c * 7 + q] = g_new[src * 7 + q];
}

template<>
void ADNeumannOutlet<D3Q7>::apply(
    ADLattice<D3Q7>& lat,
    BoundaryPhase phase,
    int
){
    // only after streaming
    if (phase != BoundaryPhase::PostStreaming) return;

    // launch config
    int n = lat.Size();
    int block = 256;
    int grid  = (n + block - 1) / block;

    // apply outlet
    k_ad_neumann<<<grid, block>>>(
        lat.d_g_new_ptr(),
        lat.P().Nx, lat.P().Ny, lat.P().Nz,
        d_mask
    );
}

template class ADNeumannOutlet<D3Q7>;
