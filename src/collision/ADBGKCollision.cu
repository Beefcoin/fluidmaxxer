//this might need work with the sourcing
#include "collision/ADBGKCollision.hpp"
#include "core/lattice/ADLattice.hpp"        
#include "core/descriptors/D3Q7.hpp"
#include "core/CudaHelpers.hpp"
#include "core/Types.hpp"
#include <cuda_runtime.h>


extern __constant__ int c7_cx[7];
extern __constant__ int c7_cy[7];
extern __constant__ int c7_cz[7];
extern __constant__ DATA_TYPE c7_w[7];

// periodic wrap / bounds check helper
__device__ __forceinline__ bool mapCoord(
    int &x, int &y, int &z,
    int Nx, int Ny, int Nz,
    bool px, bool py, bool pz
){
    // wrap if periodic, else reject
    if (px) x = (x % Nx + Nx) % Nx; else if (x < 0 || x >= Nx) return false;
    if (py) y = (y % Ny + Ny) % Ny; else if (y < 0 || y >= Ny) return false;
    if (pz) z = (z % Nz + Nz) % Nz; else if (z < 0 || z >= Nz) return false;
    return true;
}

// init g with phi0 via weights (so sum_q g = phi0)
__global__ void k_ad_init_eq_d3q7(DATA_TYPE *g, int nCells, DATA_TYPE phi0)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nCells) return;

    int base = c * 7;
    #pragma unroll
    for (int q = 0; q < 7; ++q)
        g[base + q] = phi0 * c7_w[q];
}

// BGK + pull streaming + scalar source (D3Q7)
__global__ void k_ad_bgk_stream_pull_d3q7_source(
    const DATA_TYPE* __restrict__ g,
    DATA_TYPE* __restrict__ g_new,
    DATA_TYPE* __restrict__ phi_out,
    const DATA_TYPE* __restrict__ tauField,
    const DATA_TYPE* __restrict__ ux_adv,
    const DATA_TYPE* __restrict__ uy_adv,
    const DATA_TYPE* __restrict__ uz_adv,
    const DATA_TYPE* __restrict__ Sphi,   // per-cell source
    int Nx, int Ny, int Nz,
    bool px, bool py, bool pz,
    DATA_TYPE phiMinClamp
){
    // global cell index
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    int nCells = Nx * Ny * Nz;
    if (cell >= nCells) return;

    // idx -> (x,y,z)
    int z = cell / (Nx * Ny);
    int rem = cell - z * (Nx * Ny);
    int y = rem / Nx;
    int x = rem - y * Nx;

    // local tau (field) -> omega
    DATA_TYPE tau = tauField[cell];
    DATA_TYPE omega = DATA_TYPE(1) / tau;

    // advection velocity (from coupling)
    DATA_TYPE ux = ux_adv[cell];
    DATA_TYPE uy = uy_adv[cell];
    DATA_TYPE uz = uz_adv[cell];

    // pull-stream into local gi[q]
    int base = cell * 7;
    DATA_TYPE gi[7];

    #pragma unroll
    for (int q = 0; q < 7; ++q)
    {
        int xs = x - c7_cx[q];
        int ys = y - c7_cy[q];
        int zs = z - c7_cz[q];

        // if out of domain (non-periodic), just keep local
        if (!mapCoord(xs, ys, zs, Nx, Ny, Nz, px, py, pz))
            gi[q] = g[base + q];
        else
        {
            int srcCell = xs + Nx * (ys + Ny * zs);
            gi[q] = g[srcCell * 7 + q];
        }
    }

    // phi = sum_q gi
    DATA_TYPE phi = DATA_TYPE(0);
    #pragma unroll
    for (int q = 0; q < 7; ++q)
        phi += gi[q];

    // clamp (avoid negatives / zeros)
    if (phi < phiMinClamp) phi = phiMinClamp;

    // eq for scalar+advection: geq = w * phi * (1 + (c·u)/cs^2)
    // D3Q7 cs^2 = 1/4 -> invCs2 = 4
    constexpr DATA_TYPE invCs2 = DATA_TYPE(4);

    // optional source (can be nullptr)
    DATA_TYPE src = Sphi ? Sphi[cell] : DATA_TYPE(0);

    // collide + add simple weighted source
    #pragma unroll
    for (int q = 0; q < 7; ++q)
    {
        DATA_TYPE eu =
            DATA_TYPE(c7_cx[q]) * ux +
            DATA_TYPE(c7_cy[q]) * uy +
            DATA_TYPE(c7_cz[q]) * uz;

        DATA_TYPE geq = c7_w[q] * phi * (DATA_TYPE(1) + invCs2 * eu);

        // BGK + source (simple: w_q * Sphi)
        g_new[base + q] = gi[q] + omega * (geq - gi[q]) + c7_w[q] * src;
    }

    // write phi back (optional)
    if (phi_out) phi_out[cell] = phi;
}

template <>
void ADBGKCollision<D3Q7>::initEq(ADLattice<D3Q7> &lat)
{
    // init with phi0
    const auto &p = lat.P();
    int n = p.Nx * p.Ny * p.Nz;
    int block = 256, grid = (n + block - 1) / block;

    k_ad_init_eq_d3q7<<<grid, block>>>(
        lat.d_g_ptr(),
        n,
        p.phi0
    );
    cudaCheckThrow("k_ad_init_eq_d3q7");
}

template <>
void ADBGKCollision<D3Q7>::collideStream(ADLattice<D3Q7> &lat)
{
    // collide + stream (pull) + source
    const auto &p = lat.P();
    int n = p.Nx * p.Ny * p.Nz;
    int block = 256, grid = (n + block - 1) / block;

    k_ad_bgk_stream_pull_d3q7_source<<<grid, block>>>(
        lat.d_g_ptr(),
        lat.d_g_new_ptr(),
        lat.d_phi_ptr(),
        lat.d_tau_ptr(),
        lat.d_ux_adv_ptr(), lat.d_uy_adv_ptr(), lat.d_uz_adv_ptr(),
        lat.d_Sphi_ptr(),
        p.Nx, p.Ny, p.Nz,
        p.periodicX, p.periodicY, p.periodicZ,
        p.minConcentration
    );
    cudaCheckThrow("k_ad_bgk_stream_pull_d3q7_source");
}

template class ADBGKCollision<D3Q7>;
