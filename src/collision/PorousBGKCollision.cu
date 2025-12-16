#include "collision/PorousBGKCollision.hpp"
#include "core/descriptors/D3Q19.hpp"
#include "core/CudaHelpers.hpp"
#include "core/Types.hpp"

#include <cuda_runtime.h>
#include <cstdint>

// descriptor tables
extern __constant__ int c19_cx[19];
extern __constant__ int c19_cy[19];
extern __constant__ int c19_cz[19];
extern __constant__ DATA_TYPE c19_w[19];

// ------------------------------------------------------------
// helpers
// ------------------------------------------------------------
__device__ __forceinline__ int idx3(int x, int y, int z, int Nx, int Ny)
{
    return x + Nx * (y + Ny * z);
}

__device__ __forceinline__ bool mapCoord(
    int &x, int &y, int &z,
    int Nx, int Ny, int Nz,
    bool px, bool py, bool pz
){
    if (px) x = (x % Nx + Nx) % Nx;
    else if (x < 0 || x >= Nx) return false;

    if (py) y = (y % Ny + Ny) % Ny;
    else if (y < 0 || y >= Ny) return false;

    if (pz) z = (z % Nz + Nz) % Nz;
    else if (z < 0 || z >= Nz) return false;

    return true;
}

// ------------------------------------------------------------
// init equilibrium (wie BGK: u=0)
// ------------------------------------------------------------
__global__ void k_init_eq_d3q19(
    DATA_TYPE* f,
    int nCells,
    DATA_TYPE rho0
){
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= nCells) return;

    int base = cell * 19;

    // u=0 => feq = w*rho
    #pragma unroll
    for (int q = 0; q < 19; ++q) {
        f[base + q] = c19_w[q] * rho0;
    }
}

// ------------------------------------------------------------
// Porous BGK + pull streaming
// OpenLB PorousBGKdynamics: u <- porosity * u, then plain BGK
// ------------------------------------------------------------
__global__ void k_porous_bgk_stream_pull_d3q19(
    const DATA_TYPE* __restrict__ f,
    DATA_TYPE* __restrict__ f_new,
    const DATA_TYPE* __restrict__ porosity,   // phi in [0,1], nullptr => 1
    int Nx, int Ny, int Nz,
    bool px, bool py, bool pz,
    DATA_TYPE omega,
    DATA_TYPE rho0_fallback
){
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    int nCells = Nx * Ny * Nz;
    if (cell >= nCells) return;

    int base = cell * 19;

    // coords
    int z = cell / (Nx * Ny);
    int rem = cell - z * (Nx * Ny);
    int y = rem / Nx;
    int x = rem - y * Nx;

    DATA_TYPE fi[19];

    // pull streaming (kein obstacle-reflect hier!)
    #pragma unroll
    for (int q = 0; q < 19; ++q) {
        int xs = x - c19_cx[q];
        int ys = y - c19_cy[q];
        int zs = z - c19_cz[q];

        if (!mapCoord(xs, ys, zs, Nx, Ny, Nz, px, py, pz)) {
            // out-of-domain: keep local (BCs kommen danach)
            fi[q] = f[base + q];
        } else {
            int src = idx3(xs, ys, zs, Nx, Ny);
            fi[q] = f[src * 19 + q];
        }
    }

    // moments
    DATA_TYPE rho = DATA_TYPE(0);
    DATA_TYPE jx  = DATA_TYPE(0);
    DATA_TYPE jy  = DATA_TYPE(0);
    DATA_TYPE jz  = DATA_TYPE(0);

    #pragma unroll
    for (int q = 0; q < 19; ++q) {
        const DATA_TYPE v = fi[q];
        rho += v;
        jx  += v * DATA_TYPE(c19_cx[q]);
        jy  += v * DATA_TYPE(c19_cy[q]);
        jz  += v * DATA_TYPE(c19_cz[q]);
    }

    if (!(rho > DATA_TYPE(0))) {
        rho = rho0_fallback;
        jx = jy = jz = DATA_TYPE(0);
    }

    // u = j/rho
    const DATA_TYPE invR = DATA_TYPE(1) / rho;
    DATA_TYPE ux = jx * invR;
    DATA_TYPE uy = jy * invR;
    DATA_TYPE uz = jz * invR;

    // OpenLB PorousMomentum: u *= porosity
    DATA_TYPE phi = porosity ? porosity[cell] : DATA_TYPE(1);
    if (phi < DATA_TYPE(0)) phi = DATA_TYPE(0);
    if (phi > DATA_TYPE(1)) phi = DATA_TYPE(1);

    ux *= phi;
    uy *= phi;
    uz *= phi;

    const DATA_TYPE u2 = ux*ux + uy*uy + uz*uz;

    // BGK update with feq(rho, u_porous)
    #pragma unroll
    for (int q = 0; q < 19; ++q)
    {
        const DATA_TYPE eu =
            DATA_TYPE(c19_cx[q]) * ux +
            DATA_TYPE(c19_cy[q]) * uy +
            DATA_TYPE(c19_cz[q]) * uz;

        const DATA_TYPE cu  = DATA_TYPE(3) * eu;
        const DATA_TYPE feq = c19_w[q] * rho *
            (DATA_TYPE(1) + cu + DATA_TYPE(0.5) * cu * cu - DATA_TYPE(1.5) * u2);

        f_new[base + q] = fi[q] + omega * (feq - fi[q]);
    }
}

// ------------------------------------------------------------
// class glue
// ------------------------------------------------------------
template<>
void PorousBGKCollision<D3Q19>::initEq(Lattice<D3Q19>& lat)
{
    const auto& p = lat.P();
    int n = p.Nx * p.Ny * p.Nz;

    int block = 256;
    int grid  = (n + block - 1) / block;

    k_init_eq_d3q19<<<grid, block>>>(
        lat.d_f_ptr(), n, p.rho0
    );
    cudaCheckThrow("k_init_eq_d3q19");
}

template<>
void PorousBGKCollision<D3Q19>::collideStream(Lattice<D3Q19>& lat)
{
    const auto& p = lat.P();

    DATA_TYPE tau = max(p.tau, DATA_TYPE(0.5001));
    DATA_TYPE omega = DATA_TYPE(1) / tau;

    int n = p.Nx * p.Ny * p.Nz;
    int block = 256;
    int grid  = (n + block - 1) / block;

    k_porous_bgk_stream_pull_d3q19<<<grid, block>>>(
        lat.d_f_ptr(),
        lat.d_f_new_ptr(),
        lat.d_porosity_ptr(),   // phi-field
        p.Nx, p.Ny, p.Nz,
        p.periodicX, p.periodicY, p.periodicZ,
        omega,
        p.rho0
    );
    cudaCheckThrow("k_porous_bgk_stream_pull_d3q19");
}

template class PorousBGKCollision<D3Q19>;
