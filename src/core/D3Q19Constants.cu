#include "core/descriptors/D3Q19.hpp"
#include "core/Types.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

// device constants for D3Q19
__constant__ int       c19_cx[19];
__constant__ int       c19_cy[19];
__constant__ int       c19_cz[19];
__constant__ int       c19_opp[19];
__constant__ DATA_TYPE c19_w[19];

// tiny cuda error helper
static void cudaAssert(cudaError_t code, const char* msg)
{
    if (code != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s | %s\n", msg, cudaGetErrorString(code));
        throw std::runtime_error("CUDA failure");
    }
}

// upload D3Q19 tables to constant memory
void uploadD3Q19Constants()
{
    cudaAssert(cudaMemcpyToSymbol(c19_cx,  D3Q19::cx.data(),  19 * sizeof(int)),       "cpy cx");
    cudaAssert(cudaMemcpyToSymbol(c19_cy,  D3Q19::cy.data(),  19 * sizeof(int)),       "cpy cy");
    cudaAssert(cudaMemcpyToSymbol(c19_cz,  D3Q19::cz.data(),  19 * sizeof(int)),       "cpy cz");
    cudaAssert(cudaMemcpyToSymbol(c19_opp, D3Q19::opp.data(), 19 * sizeof(int)),       "cpy opp");
    cudaAssert(cudaMemcpyToSymbol(c19_w,   D3Q19::w.data(),   19 * sizeof(DATA_TYPE)), "cpy w");
}
