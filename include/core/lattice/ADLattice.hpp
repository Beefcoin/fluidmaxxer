#pragma once
#include <vector>
#include <memory>
#include <type_traits>
#include <stdexcept>
#include <utility>

#include "core/Params.hpp"
#include "core/CudaHelpers.hpp"
#include "core/Types.hpp"
#include "core/interfaces/ADInterfaces.hpp"

// forward declaration for coupling
template <typename Descriptor>
class Lattice;

template <typename Descriptor>
class ADLattice
{
public:
    explicit ADLattice(const LBMParams &p)
        : params(p)
    {
        if (params.Nx <= 0 || params.Ny <= 0 || params.Nz <= 0)
            throw std::runtime_error("ADLattice: invalid dimensions");

        nCells = params.Nx * params.Ny * params.Nz;
        bytesG = std::size_t(nCells) * Descriptor::Q * sizeof(DATA_TYPE);

        // distribution fields
        d_g = static_cast<DATA_TYPE *>(cudaMallocBytes(bytesG));
        d_g_new = static_cast<DATA_TYPE *>(cudaMallocBytes(bytesG));

        // concentration / phi scaler field
        d_phi = static_cast<DATA_TYPE *>(cudaMallocBytes(sizeof(DATA_TYPE) * nCells));

        // Guo Style source term
        d_Sphi = static_cast<DATA_TYPE *>(cudaMallocBytes(sizeof(DATA_TYPE) * nCells));

        // advcection velocity for coupling
        d_ux_adv = static_cast<DATA_TYPE *>(cudaMallocBytes(sizeof(DATA_TYPE) * nCells));
        d_uy_adv = static_cast<DATA_TYPE *>(cudaMallocBytes(sizeof(DATA_TYPE) * nCells));
        d_uz_adv = static_cast<DATA_TYPE *>(cudaMallocBytes(sizeof(DATA_TYPE) * nCells));

        // relaxation time, this is a field to allow different diffusivities
        d_tau = static_cast<DATA_TYPE *>(cudaMallocBytes(sizeof(DATA_TYPE) * nCells));

        // initFields with memset, set everything to 0
        cudaMemsetBytes(d_phi, 0, sizeof(DATA_TYPE) * nCells);
        cudaMemsetBytes(d_Sphi, 0, sizeof(DATA_TYPE) * nCells);
        cudaMemsetBytes(d_ux_adv, 0, sizeof(DATA_TYPE) * nCells);
        cudaMemsetBytes(d_uy_adv, 0, sizeof(DATA_TYPE) * nCells);
        cudaMemsetBytes(d_uz_adv, 0, sizeof(DATA_TYPE) * nCells);

        h_phi.resize(nCells, DATA_TYPE(0));
    }

    ~ADLattice()
    {
        // free bytes for everything on destruction
        cudaFreeBytes(d_g);
        cudaFreeBytes(d_g_new);
        cudaFreeBytes(d_phi);
        cudaFreeBytes(d_Sphi);
        cudaFreeBytes(d_ux_adv);
        cudaFreeBytes(d_uy_adv);
        cudaFreeBytes(d_uz_adv);
        cudaFreeBytes(d_tau);
    }

    ADLattice(const ADLattice &) = delete;
    ADLattice &operator=(const ADLattice &) = delete;

    // Acces with getters
    const LBMParams &P() const { return params; }
    int Size() const { return nCells; }

    DATA_TYPE *d_g_ptr() { return d_g; }
    DATA_TYPE *d_g_new_ptr() { return d_g_new; }
    DATA_TYPE *d_phi_ptr() { return d_phi; }
    DATA_TYPE *d_Sphi_ptr() { return d_Sphi; }
    DATA_TYPE *d_tau_ptr() { return d_tau; }

    DATA_TYPE *d_ux_adv_ptr() { return d_ux_adv; }
    DATA_TYPE *d_uy_adv_ptr() { return d_uy_adv; }
    DATA_TYPE *d_uz_adv_ptr() { return d_uz_adv; }

    // set source to zero, i might change it so it happens automatically after stream&collide
    void zeroSource()
    {
        cudaMemsetBytes(d_Sphi, 0, sizeof(DATA_TYPE) * nCells);
    }

    // set a tau field for diffrerent diffusivities
    void setTauField(const std::vector<DATA_TYPE> &tau)
    {
        if ((int)tau.size() != nCells)
            throw std::runtime_error("ADLattice: tau field size mismatch");
        cudaMemcpyHtoD(d_tau, tau.data(), sizeof(DATA_TYPE) * nCells);
    }

    // --- COupling ---
    // here we copy the velocity from the fluid lattice from the same GPU device
    template <typename FluidDesc>
    void setAdvectionFromFluidDevice(const Lattice<FluidDesc> &fluid)
    {
        const std::size_t nBytes = std::size_t(nCells) * sizeof(DATA_TYPE);
        cudaMemcpyD2D(d_ux_adv, fluid.d_ux_ptr(), nBytes);
        cudaMemcpyD2D(d_uy_adv, fluid.d_uy_ptr(), nBytes);
        cudaMemcpyD2D(d_uz_adv, fluid.d_uz_ptr(), nBytes);
    }

    // --- BCs and Collision Operators ---
    // set collision operator
    void setCollisionOperator(std::shared_ptr<ADCollisionOperator<Descriptor>> op)
    {
        collision = std::move(op);
    }

    // add Boundary Condition
    template <typename TBoundary, typename... Args>
    TBoundary &addBoundary(Args &&...args)
    {
        static_assert(std::is_base_of_v<ADBoundaryCondition<Descriptor>, TBoundary>,
                      "AD boundary must derive from ADBoundaryCondition");
        auto ptr = std::make_unique<TBoundary>(std::forward<Args>(args)...);
        TBoundary &ref = *ptr;
        boundaries.emplace_back(std::move(ptr));
        return ref;
    }

    // -- Real LBM Stuff ---
    // init everything to eq. using the collisionOperator
    void initEquilibrium()
    {
        if (!collision)
            throw std::runtime_error("ADLattice: collision operator not set, could not initEq()!");
        collision->initEq(*this);
        //device to device copy
        cudaMemcpyHtoD(d_g_new, d_g, bytesG);
    }

    // step() function
    void step(int stepIndex)
    {
        if (!collision)
            throw std::runtime_error("ADLattice: collision operator not set, could not step()!");

        // Apply preCollision BCs
        for (auto &bc : boundaries)
            bc->apply(*this, BoundaryPhase::PreCollision, stepIndex);

        // collide&stream
        collision->collideStream(*this);

        // Apply postStreaming BCs
        for (auto &bc : boundaries)
            bc->apply(*this, BoundaryPhase::PostStreaming, stepIndex);

        // swap g, gnew
        std::swap(d_g, d_g_new);
    }

    // --- Output ---
    // download phi / concentration to host device
    void downloadPhi()
    {
        cudaMemcpyDtoH(h_phi.data(), d_phi, sizeof(DATA_TYPE) * nCells);
    }
    const std::vector<DATA_TYPE> &phiHost() const { return h_phi; }

private:
    //Parameters
    LBMParams params{};
    int nCells = 0;
    std::size_t bytesG = 0;

    // device data
    DATA_TYPE *d_g = nullptr, *d_g_new = nullptr;
    DATA_TYPE *d_phi = nullptr, *d_Sphi = nullptr;
    DATA_TYPE *d_ux_adv = nullptr, *d_uy_adv = nullptr, *d_uz_adv = nullptr;
    DATA_TYPE *d_tau = nullptr;

    // host output
    std::vector<DATA_TYPE> h_phi;

    //Collision Operators and BCs
    std::shared_ptr<ADCollisionOperator<Descriptor>> collision;
    std::vector<std::unique_ptr<ADBoundaryCondition<Descriptor>>> boundaries;
};