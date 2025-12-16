#include "boundary/ad/ADDirichletTanhPulse.hpp"
#include "core/descriptors/D3Q7.hpp"
#include "core/CudaHelpers.hpp"
#include "core/Types.hpp"
#include <cuda_runtime.h>
#include <cmath>

extern __constant__ DATA_TYPE c7_w[7];

// Dirichlet w/ smooth tanh pulse (D3Q7)
__global__ void k_ad_dirichlet_tanh_pulse_d3q7(
    DATA_TYPE* g,
    const std::uint8_t* mask,
    int nCells,
    DATA_TYPE phiIn
){
    // global cell index
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    // out of range / not in mask
    if (c >= nCells) return;
    if (!mask[c]) return;

    // base index for distributions
    int base = c * 7;

    // set g via weights so sum_q g = phiIn
    #pragma unroll
    for (int q = 0; q < 7; ++q)
        g[base + q] = phiIn * c7_w[q];
}

template<>
ADDirichletTanhPulse<D3Q7>::ADDirichletTanhPulse(
    const LBMParams& p,
    std::vector<std::uint8_t> mask,
    DATA_TYPE phi_base_,
    DATA_TYPE phi_amp_,
    int t_on_step,
    int t_off_step,
    DATA_TYPE k_step
)
: params(p),
  mask_h(std::move(mask)),
  phi_base(phi_base_),
  phi_amp(phi_amp_),
  t_on(t_on_step),
  t_off(t_off_step),
  k(k_step)
{
    // upload mask to device
    const std::size_t bytes = mask_h.size() * sizeof(std::uint8_t);
    mask_d = static_cast<std::uint8_t*>(cudaMallocBytes(bytes));
    cudaMemcpyHtoD(mask_d, mask_h.data(), bytes);
}

template<>
ADDirichletTanhPulse<D3Q7>::~ADDirichletTanhPulse()
{
    // cleanup mask
    cudaFreeBytes(mask_d);
    mask_d = nullptr;
}

template<>
void ADDirichletTanhPulse<D3Q7>::apply(ADLattice<D3Q7>& lat, BoundaryPhase phase, int step)
{
    // only pre-collision
    if (phase != BoundaryPhase::PreCollision) return;

    // pulse timing in step units
    const double tt   = static_cast<double>(step);
    const double ton  = static_cast<double>(t_on);
    const double toff = static_cast<double>(t_off);
    const double kk   = static_cast<double>(k);

    // smooth on/off
    const double pulse =
        0.5 * (std::tanh((tt - ton) / kk) - std::tanh((tt - toff) / kk));

    // inlet value = base + amp * pulse
    const DATA_TYPE phiIn =
        phi_base + static_cast<DATA_TYPE>(pulse) * phi_amp;

    // launch config
    const int nCells = params.Nx * params.Ny * params.Nz;
    const int block  = 256;
    const int grid   = (nCells + block - 1) / block;

    // write distributions on masked cells
    k_ad_dirichlet_tanh_pulse_d3q7<<<grid, block>>>(
        lat.d_g_ptr(),
        mask_d,
        nCells,
        phiIn
    );

    // check kernel launch
    cudaCheckThrow("k_ad_dirichlet_tanh_pulse_d3q7");
}

template class ADDirichletTanhPulse<D3Q7>;
