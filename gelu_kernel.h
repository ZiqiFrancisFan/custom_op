#ifndef GELU_KERNEL_
#define GELU_KERNEL_

#include <cuda.h>

void GeluKernelLauncher(const float* in, float* out, int n_elements, int n_dev, cudaStream_t stream);

#endif