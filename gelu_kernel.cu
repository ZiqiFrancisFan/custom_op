#include <cuda.h>
#include <math_constants.h>
#include <iostream>

template <typename T>
__global__ void GeluKernel(const T* in, T* out, int n_elements)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // int chunk_size = blockDim.x * gridDim.x;

    const T scale = sqrt(T(2) / CUDART_PI);

    if (gid < n_elements) {
        T x = in[gid];
        T cdf = T(1) + tanh(scale * (x + T(0.044715)) * (x * x * x));
        cdf *= T(0.5);
        out[gid] = x * cdf;
    }
}

void GeluKernelLauncher(const float* in, float* out, int n_elements, int n_dev, cudaStream_t stream)
{
    int block_size = 1024;
    int n_block = (n_elements + block_size - 1) / block_size;
    std::cout << "About to start kernel" << std::endl;
    GeluKernel<<<n_block, block_size, 0, stream>>>(in, out, n_elements);
}