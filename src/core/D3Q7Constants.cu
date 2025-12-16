#include "core/descriptors/D3Q7.hpp"
#include "core/Types.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

// device constants for D3Q7
__constant__ int       c7_cx[7];
__constant__ int       c7_cy[7];
__constant__ int       c7_cz[7];
__constant__ int       c7_opp[7];
__constant__ DATA_TYPE c7_w[7];

// tiny cuda error helper
static void cudaAssert(cudaError_t code, const char* msg)
{
    if (code != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s | %s\n", msg, cudaGetErrorString(code));
        throw std::runtime_error("CUDA failure");
    }
}

// upload D3Q7 tables to constant memory
void uploadD3Q7Constants()
{
    cudaAssert(cudaMemcpyToSymbol(c7_cx,  D3Q7::cx.data(),  7 * sizeof(int)),        "cpy D3Q7 cx");
    cudaAssert(cudaMemcpyToSymbol(c7_cy,  D3Q7::cy.data(),  7 * sizeof(int)),        "cpy D3Q7 cy");
    cudaAssert(cudaMemcpyToSymbol(c7_cz,  D3Q7::cz.data(),  7 * sizeof(int)),        "cpy D3Q7 cz");
    cudaAssert(cudaMemcpyToSymbol(c7_opp, D3Q7::opp.data(), 7 * sizeof(int)),        "cpy D3Q7 opp");
    cudaAssert(cudaMemcpyToSymbol(c7_w,   D3Q7::w.data(),   7 * sizeof(DATA_TYPE)),  "cpy D3Q7 w");
}
