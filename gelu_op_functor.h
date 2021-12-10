#ifndef GELU_OP_FUNCTOR_H_
#define GELU_OP_FUNCTOR_H_

#include <unsupported/Eigen/CXX11/Tensor>

using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;

template <typename Device, typename T> // device type and input/output type as template variables
struct GeluOpFunctor
{
    void operator()(const Device& d, const T* in, T* out, int n_elements);
};

#endif