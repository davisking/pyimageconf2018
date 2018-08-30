#ifndef SUPERFAST_CUDA_STUFF_H_
#define SUPERFAST_CUDA_STUFF_H_

#include <dlib/cuda/cuda_data_ptr.h>

using namespace dlib::cuda;

// ----------------------------------------------------------------------------------------

void cuda_add_value_to_each_element_simple (
    cuda_data_ptr<float>& img,
    const double value
);

void cuda_add_value_to_each_element_ugly (
    cuda_data_ptr<float>& img,
    const double value
);

void cuda_add_value_to_each_element (
    cuda_data_ptr<float>& img,
    const double value
);

// ----------------------------------------------------------------------------------------

void cuda_dot_product (
    cuda_data_ptr<float>& out,
    const cuda_data_ptr<float>& a,
    const cuda_data_ptr<float>& b
);

// computes out = M*v
void cuda_matrix_vector_multiply (
    cuda_data_ptr<float>& out,
    const cuda_data_ptr<float>& M,
    const cuda_data_ptr<float>& v
);

// ----------------------------------------------------------------------------------------

#endif 
