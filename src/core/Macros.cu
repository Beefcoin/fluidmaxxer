//needs overhaul
#include "core/lattice/NSLattice.hpp"
#include "core/descriptors/D3Q19.hpp"
#include "core/CudaHelpers.hpp"
#include "core/Types.hpp"
#include <cuda_runtime.h>
#include <cstdint>

extern __constant__ int c19_cx[19];
extern __constant__ int c19_cy[19];
extern __constant__ int c19_cz[19];

// 3D -> 1D idx helper
__device__ __forceinline__ int cellIndex3D(int x, int y, int z, int Nx, int Ny)
{
    return x + Nx * (y + Ny * z);
}

// compute rho + u from f (D3Q19)
__global__ void k_compute_macros_d3q19(
    const DATA_TYPE *__restrict__ f,
    DATA_TYPE *__restrict__ rho,
    DATA_TYPE *__restrict__ ux,
    DATA_TYPE *__restrict__ uy,
    DATA_TYPE *__restrict__ uz,
    int Nx, int Ny, int Nz
){
    // thread -> (x,y,z)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= Nx || y >= Ny || z >= Nz) return;

    // cell + base index
    int cell = cellIndex3D(x, y, z, Nx, Ny);
    int base = cell * 19;

    // moments
    DATA_TYPE r  = DATA_TYPE(0);
    DATA_TYPE vx = DATA_TYPE(0);
    DATA_TYPE vy = DATA_TYPE(0);
    DATA_TYPE vz = DATA_TYPE(0);

    #pragma unroll
    for (int q = 0; q < 19; ++q)
    {
        DATA_TYPE fi = f[base + q];
        r  += fi;
        vx += fi * DATA_TYPE(c19_cx[q]);
        vy += fi * DATA_TYPE(c19_cy[q]);
        vz += fi * DATA_TYPE(c19_cz[q]);
    }

    // u = m / rho (avoid div0)
    DATA_TYPE invR = (r > DATA_TYPE(0)) ? (DATA_TYPE(1) / r) : DATA_TYPE(0);

    rho[cell] = r;
    ux[cell]  = vx * invR;
    uy[cell]  = vy * invR;
    uz[cell]  = vz * invR;
}

template <>
void Lattice<D3Q19>::downloadMacroscopic()
{
    const auto &p = this->P();
    const std::size_t nBytes = (std::size_t)this->Size() * sizeof(DATA_TYPE);

    // lazy alloc on device
    if (!d_rho) d_rho = (DATA_TYPE *)cudaMallocBytes(nBytes);
    if (!d_ux)  d_ux  = (DATA_TYPE *)cudaMallocBytes(nBytes);
    if (!d_uy)  d_uy  = (DATA_TYPE *)cudaMallocBytes(nBytes);
    if (!d_uz)  d_uz  = (DATA_TYPE *)cudaMallocBytes(nBytes);

    // 3D launch config
    dim3 block(8, 8, 8);
    dim3 grid((p.Nx + block.x - 1) / block.x,
              (p.Ny + block.y - 1) / block.y,
              (p.Nz + block.z - 1) / block.z);

    // compute on device
    k_compute_macros_d3q19<<<grid, block>>>(
        d_f_ptr(), d_rho, d_ux, d_uy, d_uz,
        p.Nx, p.Ny, p.Nz
    );
    cudaCheckThrow("k_compute_macros_d3q19");

    // download to host
    cudaMemcpyDtoH(h_rho.data(), d_rho, nBytes);
    cudaMemcpyDtoH(h_ux.data(),  d_ux,  nBytes);
    cudaMemcpyDtoH(h_uy.data(),  d_uy,  nBytes);
    cudaMemcpyDtoH(h_uz.data(),  d_uz,  nBytes);
}

template <>
void Lattice<D3Q19>::computeMacroscopicDevice()
{
    const auto &p = this->P();
    const std::size_t nBytes = (std::size_t)this->Size() * sizeof(DATA_TYPE);

    // lazy alloc on device
    if (!d_rho) d_rho = (DATA_TYPE *)cudaMallocBytes(nBytes);
    if (!d_ux)  d_ux  = (DATA_TYPE *)cudaMallocBytes(nBytes);
    if (!d_uy)  d_uy  = (DATA_TYPE *)cudaMallocBytes(nBytes);
    if (!d_uz)  d_uz  = (DATA_TYPE *)cudaMallocBytes(nBytes);

    // 3D launch config
    dim3 block(8, 8, 8);
    dim3 grid((p.Nx + block.x - 1) / block.x,
              (p.Ny + block.y - 1) / block.y,
              (p.Nz + block.z - 1) / block.z);

    // compute only (no download)
    k_compute_macros_d3q19<<<grid, block>>>(
        d_f_ptr(), d_rho, d_ux, d_uy, d_uz,
        p.Nx, p.Ny, p.Nz
    );
    cudaCheckThrow("computeMacroscopicDevice");
}

template <>
void Lattice<D3Q19>::downloadObstacle()
{
    // just pull obstacle mask
    const std::size_t nBytes = (std::size_t)this->Size() * sizeof(std::uint8_t);
    cudaMemcpyDtoH(h_obstacle.data(), d_obstacle_ptr(), nBytes);
}

template class Lattice<D3Q19>;
