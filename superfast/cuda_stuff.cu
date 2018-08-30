#include <dlib/cuda/cuda_utils.h>
#include <dlib/cuda/cuda_data_ptr.h>
#include "cuda_stuff.h"

// ----------------------------------------------------------------------------------------

// __global__ is a CUDA keyword that means "this function runs on the GPU".  People call
// such functions "kernels".
__global__ void kernel_add_value_to_each_element_simple(float* data, const float value, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;    
    if (i < n)
        data[i] += value;
}

void cuda_add_value_to_each_element_simple (
    cuda_data_ptr<float>& img,
    const double value
)
{
    kernel_add_value_to_each_element_simple<<<4096,256>>>(img, value, img.size());
}

// ----------------------------------------------------------------------------------------

__global__ void kernel_add_value_to_each_element_ugly(float* data, const float value, size_t n)
{
    const auto num_cuda_threads = blockDim.x * gridDim.x;
    const auto thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // this is called a "grid stride loop" and allows our kernel to efficiently handle
    // arrays of any length, regardless of how many cuda threads we have running.
    for (auto i = thread_idx; i < n; i += num_cuda_threads)
    {
        data[i] += value;
    }
}

void cuda_add_value_to_each_element_ugly (
    cuda_data_ptr<float>& img,
    const double value
)
{
    kernel_add_value_to_each_element_ugly<<<4096,256>>>(img, value, img.size());
}

// ----------------------------------------------------------------------------------------

__global__ void kernel_add_value_to_each_element(float* data, const float value, size_t n)
{
    // This is a nice C++11 way to write the same grid stride loop as above.
    for (auto i : grid_stride_range(0, n))
    {
        data[i] += value;
    }
}

void cuda_add_value_to_each_element (
    cuda_data_ptr<float>& img,
    const double value
)
{
    launch_kernel(kernel_add_value_to_each_element, img, value, img.size());
}

// ----------------------------------------------------------------------------------------

__global__ void kernel_set_to_0(float* out, size_t n)
{
    for (auto i : grid_stride_range(0, n))
        out[i] = 0;
}

__global__ void kernel_dot_product(float* out, const float* a, const float* b, size_t n)
{
    // Parallel sum everything into local temp variables.
    float temp = 0;
    for (auto i : grid_stride_range(0, n))
        temp += a[i]*b[i];
    
    // Each CUDA thread has a temp variable that contains a partial sum.  We need to get
    // them to all add together to out, but we can't just do:
    //   *out += temp
    // because it's illegal for multiple threads to write to the same memory at the same
    // time (as you might expect).  To do this, you need to use special atomic commands
    // that force the appropriate synchronization:
    warp_reduce_atomic_add(*out, temp); // do *out += temp, but legally.
    // There is a good article that goes into "warp reduce atomic" stuff here: https://devblogs.nvidia.com/faster-parallel-reductions-kepler/
}

void cuda_dot_product (
    cuda_data_ptr<float>& out,
    const cuda_data_ptr<float>& a,
    const cuda_data_ptr<float>& b
)
{
    DLIB_CASSERT(a.size() == b.size());
    DLIB_CASSERT(out.size() == 1);

    // tell cuda not to launch more than out.size() threads so we don't waste resources.
    launch_kernel(kernel_set_to_0, max_jobs(out.size()), out, out.size());
    // run the dot product kernel
    launch_kernel(kernel_dot_product, out, a, b, a.size());
}

// ----------------------------------------------------------------------------------------

__global__ void kernel_matrix_vector_multiply (float* out, const float* M, const float* v, size_t nr, size_t nc)
{
    // initialize out to 0
    for (auto r : grid_stride_range_y(0, nr))
        for (auto c : grid_stride_range(0, 1))
            out[r] = 0;

    __syncthreads(); // synchronize threads in block so we don't start the next bit until out is really 0.

    for (auto r : grid_stride_range_y(0, nr))
    {
        float temp = 0;
        for (auto c : grid_stride_range(0, nc))
            temp += M[r*nc+c]*v[c];

        // store the sum into out[r]
        warp_reduce_atomic_add(out[r], temp);
    }
}

// computes out = M*v
void cuda_matrix_vector_multiply (
    cuda_data_ptr<float>& out,
    const cuda_data_ptr<float>& M,
    const cuda_data_ptr<float>& v
)
{
    DLIB_CASSERT(M.size() == out.size()*v.size());

    const auto nr = out.size();
    const auto nc = v.size();
    launch_kernel(kernel_matrix_vector_multiply, max_jobs(nc,nr), out, M, v, nr, nc);
}

// ----------------------------------------------------------------------------------------

/*
    talk about cudaSetDevice()

    See also:
    https://devblogs.nvidia.com/even-easier-introduction-cuda/


    I also like the book:
    Professional CUDA C Programming by John Cheng, Max Grossman, and Ty McKercher
*/
