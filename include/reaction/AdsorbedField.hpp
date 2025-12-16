#pragma once
#include <vector>
#include <stdexcept>
#include "core/Params.hpp"
#include "core/CudaHelpers.hpp"

class AdsorbedField
{
public:
    explicit AdsorbedField(const LBMParams& p)
        : p_(p)
    {
        // number of cells
        n_ = p_.Nx * p_.Ny * p_.Nz;

        // adsorbed amount q on device
        d_q_ = (DATA_TYPE*)cudaMallocBytes(sizeof(DATA_TYPE) * n_);

        // init with memset, set everything to 0
        cudaMemsetBytes(d_q_, 0, sizeof(DATA_TYPE) * n_);

        // host mirror (for output / init)
        h_q_.assign(n_, DATA_TYPE(0));
    }

    ~AdsorbedField()
    {
        // free device memory on destruction
        cudaFreeBytes(d_q_);
    }

    // quick device getters
    DATA_TYPE* d_q() { return d_q_; }
    const DATA_TYPE* d_q() const { return d_q_; }

    // pull q from device to host
    void download()
    {
        cudaMemcpyDtoH(h_q_.data(), d_q_, sizeof(DATA_TYPE) * n_);
    }

    // push q from host to device
    void upload()
    {
        cudaMemcpyHtoD(d_q_, h_q_.data(), sizeof(DATA_TYPE) * n_);
    }

    // host access
    std::vector<DATA_TYPE>& host() { return h_q_; }
    const std::vector<DATA_TYPE>& host() const { return h_q_; }

private:
    // params
    LBMParams p_;
    int n_ = 0;

    // device,host storage
    DATA_TYPE* d_q_ = nullptr;
    std::vector<DATA_TYPE> h_q_;
};
