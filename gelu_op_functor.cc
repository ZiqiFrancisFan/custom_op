#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <iostream>
#include "gelu_op_functor.h"
#include "gelu_kernel.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cuda.h>
#include <stdio.h>

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

#if GOOGLE_CUDA

template <typename T>
struct GeluOpFunctor<GPUDevice, T>
{
    GeluOpFunctor() // constructor of GeluOpFunctor
    {
        int device;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&num_devices_, cudaDevAttrMultiProcessorCount, device);
    }

    void operator()(const GPUDevice& d, const T* in, T* out, int n_elements) // true execution of the functor
    {
        std::cout << "in operator()" << std::endl;

        T *in_h, *out_h;
        in_h = new T[n_elements];
        out_h = new T[n_elements];

        cudaError_t status;
        status = cudaMemcpy(in_h, in, n_elements*sizeof(T), cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) {
            std::cout << "Something bad happened on line " << __LINE__ << " in file " << __FILE__ << std::endl;
        }

        for (int i = 0; i < n_elements; i++) {
            std::cout << "The " << i << "th element is " << in_h[i] << std::endl; 
        }

        GeluKernelLauncher(in, out, n_elements, num_devices_, d.stream());

        status = cudaMemcpy(out_h, out, n_elements*sizeof(T), cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) {
            std::cout << "Something bad happened on line " << __LINE__ << " in file " << __FILE__ << std::endl;
        }

        for (int i = 0; i < n_elements; i++) {
            printf("The %dth element is %f\n", i, out_h[i]); 
        }

        delete[] in_h;
        delete[] out_h;
    }

    int num_devices_;
};

template struct GeluOpFunctor<GPUDevice, float>;

#endif

#endif