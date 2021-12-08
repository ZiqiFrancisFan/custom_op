#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "gelu_op.h"
#include "gelu_kernel.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

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
        GeluKernelLauncher(in, out, n_elements, num_devices_, d.stream());
    }

    int num_devices_;
};

template struct GeluOpFunctor<GPUDevice, float>;

#endif