#ifndef GELU_OP_H_
#define GELU_OP_H_

template <typename Device, typename T> // device type and input/output type as template variables
struct GeluOpFunctor
{
    void operator()(const Device& d, const T* in, T* out, int n_elements);
};

#if GOOGLE_CUDA

template <typename T>
struct GeluOpFunctor<Eigen::GpuDevice, T>
{
    void operator()(const Eigen::GpuDevice& d, const T* in, T* out, int n_elements);
};

#endif

#endif