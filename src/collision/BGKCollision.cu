//needs work because of boundary condition
#include "collision/BGKCollision.hpp"
#include "core/descriptors/D3Q19.hpp"
#include "core/CudaHelpers.hpp"
#include "core/Types.hpp"

#include <cstdint>
#include <cuda_runtime.h>

// const tables (cx/cy/cz + weights)
extern __constant__ int c19_cx[19];
extern __constant__ int c19_cy[19];
extern __constant__ int c19_cz[19];
extern __constant__ DATA_TYPE c19_w[19];

// opp map (for bounceback)
extern __constant__ int c19_opp[19];

// 3D -> 1D idx
__device__ __forceinline__ int idx3(int x, int y, int z, int Nx, int Ny)
{
    return x + Nx * (y + Ny * z);
}

// periodic wrap / bounds check
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

// init f with rho0, u=0 (eq fill)
__global__ void k_init_eq_d3q19_1d(DATA_TYPE* f, int nCells, DATA_TYPE rho0)
{
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= nCells) return;

    int base = cell * 19;

    // u=0 -> eu=0, u2=0
    const DATA_TYPE ux = DATA_TYPE(0);
    const DATA_TYPE uy = DATA_TYPE(0);
    const DATA_TYPE uz = DATA_TYPE(0);
    const DATA_TYPE u2 = DATA_TYPE(0);

    #pragma unroll
    for (int q = 0; q < 19; ++q)
    {
        const DATA_TYPE eu =
            DATA_TYPE(c19_cx[q]) * ux +
            DATA_TYPE(c19_cy[q]) * uy +
            DATA_TYPE(c19_cz[q]) * uz;

        const DATA_TYPE cu  = DATA_TYPE(3) * eu;
        const DATA_TYPE feq = c19_w[q] * rho0 *
                              (DATA_TYPE(1) + cu + DATA_TYPE(0.5) * cu * cu - DATA_TYPE(1.5) * u2);

        f[base + q] = feq;
    }
}

// BGK + pull streaming (obstacles inside)
__global__ void k_bgk_stream_pull_d3q19(
    const DATA_TYPE* __restrict__ f,
    DATA_TYPE* __restrict__ f_new,
    const std::uint8_t* __restrict__ obstacle,
    int Nx, int Ny, int Nz,
    bool px, bool py, bool pz,
    DATA_TYPE omega,
    DATA_TYPE rho0_fallback
){
    // global cell index
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    int nCells = Nx * Ny * Nz;
    if (cell >= nCells) return;

    const int base = cell * 19;

    // solid cell: on-site bounceback
    if (obstacle && obstacle[cell])
    {
        #pragma unroll
        for (int q = 0; q < 19; ++q)
            f_new[base + q] = f[base + c19_opp[q]];
        return;
    }

    // idx -> (x,y,z)
    int z = cell / (Nx * Ny);
    int rem = cell - z * (Nx * Ny);
    int y = rem / Nx;
    int x = rem - y * Nx;

    // pulled distributions
    DATA_TYPE fi[19];

    #pragma unroll
    for (int q = 0; q < 19; ++q)
    {
        int xs = x - c19_cx[q];
        int ys = y - c19_cy[q];
        int zs = z - c19_cz[q];

        // out-of-domain -> keep local
        if (!mapCoord(xs, ys, zs, Nx, Ny, Nz, px, py, pz))
        {
            fi[q] = f[base + q];
        }
        else
        {
            int src = idx3(xs, ys, zs, Nx, Ny);

            // pulling from solid -> reflect
            if (obstacle && obstacle[src])
                fi[q] = f[base + c19_opp[q]];
            else
                fi[q] = f[src * 19 + q];
        }
    }

    // moments from fi
    DATA_TYPE rho = DATA_TYPE(0);
    DATA_TYPE ux  = DATA_TYPE(0);
    DATA_TYPE uy  = DATA_TYPE(0);
    DATA_TYPE uz  = DATA_TYPE(0);

    #pragma unroll
    for (int q = 0; q < 19; ++q)
    {
        const DATA_TYPE v = fi[q];
        rho += v;
        ux  += v * DATA_TYPE(c19_cx[q]);
        uy  += v * DATA_TYPE(c19_cy[q]);
        uz  += v * DATA_TYPE(c19_cz[q]);
    }

    // fallback if rho is bad
    if (!(rho > DATA_TYPE(0)))
    {
        rho = rho0_fallback;
        ux = uy = uz = DATA_TYPE(0);
    }
    else
    {
        const DATA_TYPE invR = DATA_TYPE(1) / rho;
        ux *= invR; uy *= invR; uz *= invR;
    }

    const DATA_TYPE u2 = ux * ux + uy * uy + uz * uz;

    // BGK update
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

template<>
void BGKCollision<D3Q19>::initEq(Lattice<D3Q19>& lat)
{
    // init eq
    const auto& p = lat.P();
    const int n = p.Nx * p.Ny * p.Nz;

    const int block = 256;
    const int grid  = (n + block - 1) / block;

    k_init_eq_d3q19_1d<<<grid, block>>>(
        lat.d_f_ptr(),
        n,
        p.rho0
    );
    cudaCheckThrow("k_init_eq_d3q19_1d");
}

template<>
void BGKCollision<D3Q19>::collideStream(Lattice<D3Q19>& lat)
{
    const auto& p = lat.P();

    // omega with tiny clamp
    DATA_TYPE tau = p.tau;
    if (tau < DATA_TYPE(0.5001)) tau = DATA_TYPE(0.5001);
    const DATA_TYPE omega = DATA_TYPE(1) / tau;

    // launch config
    const int n = p.Nx * p.Ny * p.Nz;
    const int block = 256;
    const int grid  = (n + block - 1) / block;

    // collide + stream
    k_bgk_stream_pull_d3q19<<<grid, block>>>(
        lat.d_f_ptr(),
        lat.d_f_new_ptr(),
        lat.d_obstacle_ptr(),
        p.Nx, p.Ny, p.Nz,
        p.periodicX, p.periodicY, p.periodicZ,
        omega,
        p.rho0
    );
    cudaCheckThrow("k_bgk_stream_pull_d3q19");
}

template class BGKCollision<D3Q19>;
