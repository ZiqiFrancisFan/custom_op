#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "gelu_op_functor.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;

REGISTER_OP("GeluOp")
    .Attr("T: numbertype")
    .Input("in: T")
    .Output("out: T")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

template <typename Device, typename T>
class GeluOp : public OpKernel {
public:
    explicit GeluOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) {

        const Tensor& input_tensor = context->input(0);

        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));

        const T* input_ptr = input_tensor.flat<T>().data();
        T* output_ptr = output_tensor->flat<T>().data();
        int num_elements = input_tensor.NumElements();

        functor_(context->eigen_device<Device>(), input_ptr, output_ptr, num_elements);
    }

private:
    GeluOpFunctor<Device, T> functor_;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T) \ 

extern template class ExampleFunctor<GPUDevice, T>;

REGISTER_KERNEL_BUILDER( \
    Name("GeluOp").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
    GeluOp<GPUDevice, T>);

REGISTER_GPU(float);

// REGISTER_KERNEL_BUILDER(Name("GeluOp").Device(DEVICE_GPU).TypeConstraint<float>("T"), GeluOp<GPUDevice, float>);
// REGISTER_KERNEL_BUILDER(Name("GeluOp").Device(DEVICE_GPU).TypeConstraint<int16>("T"), GeluOp<GPUDevice, int16>);
//REGISTER_GPU(double);

#undef REGISTER_GPU

#endif
