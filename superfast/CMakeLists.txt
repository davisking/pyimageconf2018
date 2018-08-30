
CMAKE_MINIMUM_REQUIRED(VERSION 3.11)
project(superfast LANGUAGES CXX CUDA)

set (CMAKE_CXX_STANDARD 14)

# Set this to disable link time optimization. The only reason for
# doing this to make the compile faster which is nice when developing
# new modules.
set(PYBIND11_LTO_CXX_FLAGS "")


# pull in dependencies
add_subdirectory(../dlib/dlib/external/pybind11 pybind11_build)
add_subdirectory(../dlib dlib_build)



# build our python extension
pybind11_add_module(superfast 
   main.cpp 
   cuda_stuff.cu
   )
target_link_libraries(superfast PRIVATE dlib::dlib )


# There is a good article about setting up CMake to build CUDA applications
# here: https://devblogs.nvidia.com/building-cuda-applications-cmake/

