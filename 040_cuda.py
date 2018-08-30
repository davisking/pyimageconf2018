import sys
sys.path = ['./superfast/build'] + sys.path

import superfast
import numpy as np
import timeit



arr = np.full((10,1), 3.0, dtype='float32')

print("USING NUMPY")
print(arr)
# This is the version that copies to the GPU, adds 2, then copies back to the CPU.
superfast.cuda_add_value_to_each_element(arr, 2)
print(arr)



print("\nUSING CUDA")
# But really you want to avoid copying data between CPU and GPU.  So typically
# you will create an object to represent your GPU array:
cuda_arr = superfast.cuda_data(arr)

# Then you can do all the cuda stuff you want to do, generally many things:
superfast.cuda_add_value_to_each_element(cuda_arr, 2)
superfast.cuda_add_value_to_each_element(cuda_arr, 2)
superfast.cuda_add_value_to_each_element(cuda_arr, 2)
# And then you copy back to the CPU
arr = cuda_arr.to_numpy()
print("cuda data: \n", arr)






# Now lets call our dot product and matrix vector multiply we wrote:

# First, the dot product:
tmp = superfast.cuda_data(1)
superfast.cuda_dot_product(tmp, cuda_arr, cuda_arr)
tmp2 = cuda_arr.to_numpy()
print("\n\nnumpy dot {}, cuda dot {}".format(np.sum(tmp2*tmp2),  tmp.to_numpy()))


# Now the matrix vector multiply with a small matrix so we can look at the outputs.
M = np.matrix(np.random.randn(2,3).astype('float32'))
v = np.matrix(np.random.randn(3,1).astype('float32'))
out = M*v
print("M*v done via numpy gives: \n", out)

cudaM = superfast.cuda_data(np.asarray(M))
cudav = superfast.cuda_data(np.asarray(v))
cudaout = superfast.cuda_data(2)

superfast.cuda_matrix_vector_multiply(cudaout, cudaM, cudav)
out2 = cudaout.to_numpy()
print("\nM*v done via cuda gives: \n", out2)


err = np.sum(np.square(out-out2))
print("difference between cuda and numpy: ", err)









# Now try M*v with a big matrix and see how it performs
M = np.matrix(np.random.randn(6000,6000).astype('float32'))
v = np.matrix(np.random.randn(6000,1).astype('float32'))

cudaM = superfast.cuda_data(np.asarray(M))
cudav = superfast.cuda_data(np.asarray(v))
cudaout = superfast.cuda_data(6000)


# Do M*v 1000 times on the CPU and see how long it takes.
start = timeit.default_timer()
for i in range(0,1000):
    out = M*v
stop = timeit.default_timer()
print("\n\nnumpy time to do M*v 1000 times: ", stop-start)

# Do M*v 1000 times on the GPU and see how long it takes.
cudastart = timeit.default_timer()
for i in range(0,1000):
    superfast.cuda_matrix_vector_multiply(cudaout, cudaM, cudav)
#superfast.cuda_device_synchronize(0)
out2 = cudaout.to_numpy()
cudastop = timeit.default_timer()

print("cuda time to do M*v 1000 times: ", cudastop-cudastart)
print("speedup over numpy: ", (stop-start)/(cudastop-cudastart))

# Also check that we got basically the same results:
err = np.sum(np.square(out-out2))
print("difference between cuda and numpy: ", err)

