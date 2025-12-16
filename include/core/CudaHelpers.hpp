#pragma once
#include <cstddef>

//helper for check error and throw exception
void cudaCheckThrow(const char* msg);

//allocate device memory
void* cudaMallocBytes(std::size_t bytes);

//free device memory
void  cudaFreeBytes(void* p);

//copy data from Host to Device
void  cudaMemcpyHtoD(void* d, const void* h, std::size_t bytes);

//copy data from Device to Host
void  cudaMemcpyDtoH(void* h, const void* d, std::size_t bytes);

//copy data from Device to Device
void cudaMemcpyD2D(void* dst, const void* src, std::size_t bytes);

//set data (MemSet Wrapper)
void  cudaMemsetBytes(void* d, int value, std::size_t bytes);

