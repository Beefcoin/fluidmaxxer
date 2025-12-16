#include "core/CudaHelpers.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

// helper for check error and throw exception
static void cudaAssert(cudaError_t code, const char* msg)
{
    if (code != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s | %s\n", msg, cudaGetErrorString(code));
        throw std::runtime_error("CUDA failure");
    }
}

// helper for check error and throw exception
void cudaCheckThrow(const char* msg)
{
    cudaAssert(cudaGetLastError(), msg);
    cudaAssert(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

// allocate device memory
void* cudaMallocBytes(std::size_t bytes)
{
    void* p = nullptr;
    cudaAssert(cudaMalloc(&p, bytes), "cudaMalloc");
    return p;
}

// free device memory
void cudaFreeBytes(void* p)
{
    if (p)
        cudaAssert(cudaFree(p), "cudaFree");
}

// copy data from Host to Device
void cudaMemcpyHtoD(void* d, const void* h, std::size_t bytes)
{
    cudaAssert(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice), "cudaMemcpyHtoD");
}

// copy data from Device to Host
void cudaMemcpyDtoH(void* h, const void* d, std::size_t bytes)
{
    cudaAssert(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost), "cudaMemcpyDtoH");
}

// copy data from Device to Device
void cudaMemcpyD2D(void* dst, const void* src, std::size_t bytes)
{
    cudaAssert(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice), "cudaMemcpyD2D");
}

// set data (MemSet Wrapper)
void cudaMemsetBytes(void* d, int value, std::size_t bytes)
{
    cudaAssert(cudaMemset(d, value, bytes), "cudaMemset");
}
