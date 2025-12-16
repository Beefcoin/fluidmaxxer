#pragma once
#include <vector>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <stdexcept>
#include <utility>

#include "core/Params.hpp"
#include "core/interfaces/NSInterfaces.hpp"
#include "core/CudaHelpers.hpp"
#include "core/Types.hpp"

template <typename Descriptor>
class Lattice
{
public:
    explicit Lattice(const LBMParams &p)
        : params(p)
    {
        if (params.Nx <= 0 || params.Ny <= 0 || params.Nz <= 0)
            throw std::runtime_error("NSLattice: invalid dimensions");

        // Size Calculations for mem allocation
        nCells = params.Nx * params.Ny * params.Nz;
        bytesF = std::size_t(nCells) * Descriptor::Q * sizeof(DATA_TYPE);
        bytesCells = std::size_t(nCells) * sizeof(DATA_TYPE);
        bytesMask = std::size_t(nCells) * sizeof(std::uint8_t);

        // distribution fields
        d_f = static_cast<DATA_TYPE *>(cudaMallocBytes(bytesF));
        d_f_new = static_cast<DATA_TYPE *>(cudaMallocBytes(bytesF));

        // obstacle mask field
        d_obstacle = static_cast<std::uint8_t *>(cudaMallocBytes(bytesMask));
        cudaMemsetBytes(d_obstacle, 0, bytesMask);

        // host side output buffers
        h_rho.assign(nCells, static_cast<DATA_TYPE>(params.rho0));
        h_ux.assign(nCells, DATA_TYPE(0));
        h_uy.assign(nCells, DATA_TYPE(0));
        h_uz.assign(nCells, DATA_TYPE(0));
        h_obstacle.assign(nCells, 0);

        // device macro buffers are lazy
        d_rho = d_ux = d_uy = d_uz = nullptr;
    }

    ~Lattice()
    {
        // destruct and clear everything
        cudaFreeBytes(d_f);
        cudaFreeBytes(d_f_new);
        cudaFreeBytes(d_obstacle);

        cudaFreeBytes(d_rho);
        cudaFreeBytes(d_ux);
        cudaFreeBytes(d_uy);
        cudaFreeBytes(d_uz);

        d_f = d_f_new = nullptr;
        d_obstacle = nullptr;
        d_rho = d_ux = d_uy = d_uz = nullptr;
    }

    Lattice(const Lattice &) = delete;
    Lattice &operator=(const Lattice &) = delete;

    // indexing used by VTIWriter
    inline int cellIndex(int x, int y, int z) const
    {
        return x + params.Nx * (y + params.Ny * z);
    }

    // Parameter getters
    const LBMParams &P() const { return params; }
    int Size() const { return nCells; }

    // Device Pointer getters
    DATA_TYPE *d_f_ptr() { return d_f; }
    const DATA_TYPE *d_f_ptr() const { return d_f; }

    DATA_TYPE *d_f_new_ptr() { return d_f_new; }
    const DATA_TYPE *d_f_new_ptr() const { return d_f_new; }

    std::uint8_t *d_obstacle_ptr() { return d_obstacle; }
    const std::uint8_t *d_obstacle_ptr() const { return d_obstacle; }

    // Macroscopic value getters (allocated on demand by computeMacroscopicDevice)
    DATA_TYPE *d_rho_ptr() { return d_rho; }
    const DATA_TYPE *d_rho_ptr() const { return d_rho; }

    DATA_TYPE *d_ux_ptr() { return d_ux; }
    const DATA_TYPE *d_ux_ptr() const { return d_ux; }

    DATA_TYPE *d_uy_ptr() { return d_uy; }
    const DATA_TYPE *d_uy_ptr() const { return d_uy; }

    DATA_TYPE *d_uz_ptr() { return d_uz; }
    const DATA_TYPE *d_uz_ptr() const { return d_uz; }

    // set obstacle mask, might be used for BounceBack BC or Collision (kernel BOunceback or porosity)
    void setObstacleMask(const std::vector<std::uint8_t> &mask)
    {
        if ((int)mask.size() != nCells)
            throw std::runtime_error("NSLattice::setObstacleMask: mask size mismatch");
        cudaMemcpyHtoD(d_obstacle, mask.data(), bytesMask);
    }

    // set collision operator
    void setCollisionOperator(std::shared_ptr<NSCollisionOperator<Descriptor>> op)
    {
        collision = std::move(op);
    }

    // Add boundary condition
    template <typename TBoundary, typename... Args>
    TBoundary &addBoundary(Args &&...args)
    {
        static_assert(std::is_base_of_v<NSBoundaryCondition<Descriptor>, TBoundary>,
                      "addBoundary: TBoundary must derive from NSBoundaryCondition<Descriptor>");
        auto ptr = std::make_unique<TBoundary>(std::forward<Args>(args)...);
        TBoundary &ref = *ptr;
        boundaries.emplace_back(std::move(ptr));
        return ref;
    }

    // init everything to eq. using the collisionOperator
    void initEquilibrium()
    {
        if (!collision)
            throw std::runtime_error("NSLattice::initEquilibrium: collision operator not set");

        collision->initEq(*this);
        // device to device copy
        cudaMemcpyD2D(d_f_new, d_f, bytesF);
    }

    // step() function
    void step(int stepIndex)
    {
        if (!collision)
            throw std::runtime_error("NSLattice: collision operator not set, could not step()!");

        // Apply preCollision BCs
        for (auto &bc : boundaries)
            bc->apply(*this, BoundaryPhase::PreCollision, stepIndex);

        // carry copy if kernel doesnt write all entries ( i think it can be removed )
        cudaMemcpyD2D(d_f_new, d_f, bytesF);

        // collide&stream
        collision->collideStream(*this);

        // Apply postStreaming BCs
        for (auto &bc : boundaries)
            bc->apply(*this, BoundaryPhase::PostStreaming, stepIndex);

        // swap f, fnew
        std::swap(d_f, d_f_new);
    }

    // --- Output ---
    // downlaod Macroscopics and obstacle mask
    void downloadMacroscopic();
    void downloadObstacle();

    // getters for macrosdopcic host data
    const std::vector<DATA_TYPE> &rhoHost() const { return h_rho; }
    const std::vector<DATA_TYPE> &uxHost() const { return h_ux; }
    const std::vector<DATA_TYPE> &uyHost() const { return h_uy; }
    const std::vector<DATA_TYPE> &uzHost() const { return h_uz; }
    const std::vector<std::uint8_t> &obstacleHost() const { return h_obstacle; }

    // compute device macros (implemented in src/core/Macros.cu)
    void computeMacroscopicDevice();

private:
    LBMParams params{};
    int nCells = 0;

    std::size_t bytesF = 0;
    std::size_t bytesCells = 0;
    std::size_t bytesMask = 0;

    // distribution
    DATA_TYPE *d_f = nullptr;
    DATA_TYPE *d_f_new = nullptr;

    // obstacle mask
    std::uint8_t *d_obstacle = nullptr;

    // device macros
    DATA_TYPE *d_rho = nullptr;
    DATA_TYPE *d_ux = nullptr;
    DATA_TYPE *d_uy = nullptr;
    DATA_TYPE *d_uz = nullptr;

    // host buffers for output
    std::vector<DATA_TYPE> h_rho, h_ux, h_uy, h_uz;
    std::vector<std::uint8_t> h_obstacle;

    // Bcs and coll. operator
    std::shared_ptr<NSCollisionOperator<Descriptor>> collision;
    std::vector<std::unique_ptr<NSBoundaryCondition<Descriptor>>> boundaries;

    // allow some kernels direct access, was for testing, can be removed soon
    template <typename D>
    friend class BGKCollision;
    template <typename D>
    friend class BounceBack;
};