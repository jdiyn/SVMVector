#include <boost/compute.hpp>

// set up alignment of the struct to opencl compatible
#ifdef _MSC_VER
#  define ALIGN_16 __declspec(align(16))
#else
#  define ALIGN_16 __attribute__((aligned(16)))
#endif

// if struct contains a float4, the device compiler expects that field to start on a 16?byte boundary
ALIGN_16 struct TestStruct {
    boost::compute::int_    type;
    boost::compute::float_  radius;
    boost::compute::float4_ dims;

};

// A trivial kernel that prints + modifies some fields in the struct array.
static const char SVM_STRUCT_KERNEL_SRC[] = R"CLC(
    // declare the stuct in alignment with the cpp verison
    // if a different struct produces errors, you might need to still introduce
    // explicit padding for alignment
    typedef struct __attribute__((aligned(16))) {
        int    type;    // offset 0
        float  radius;  // offset 4
        float4 dims;    // offset 16..31 => total 32
    } TestStruct;

__kernel void StructKernel(
    __global TestStruct* arr, // SVM pointer to array of TestStruct
    int n,
    int debug
){
    int idx = get_global_id(0);
    if(idx >= n) return;

    // Print out sample data but only first few, to minimise console spam
    if(debug != 0 && idx < 4){
        printf("[KERNEL DEBUG] Sample Data: idx=%d => type=%d, radius=%.6f, dims=(%.3f,%.3f,%.3f,%.3f)\n",
            idx,
            arr[idx].type,
            arr[idx].radius,
            arr[idx].dims.x,
            arr[idx].dims.y,
            arr[idx].dims.z,
            arr[idx].dims.w
        );
    }

    // modify one field (print out only one description)
    if(idx == 0) {
    printf("Kernel will add 1.0 to radius of each element");
    }
    arr[idx].radius += 1.0f;
}
)CLC";
#pragma once
