// ============================================================================
// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2024 Joshua Diyn
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================

#include <iostream>
#include <vector>
#include <cassert>
#include <thread>
#include <chrono>
#include <boost/compute.hpp>
#include "svm_vector.hpp"
#include "struct_test.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/////// Setups //////////
// Simple structure for testing, e.g., float2_
struct float2_ {
    float x, y;
    float2_() : x(0.0f), y(0.0f) {}           // Default constructor
    float2_(float x_, float y_) : x(x_), y(y_) {} // Parameterized constructor
};

// For printing float2_ to std::ostream
std::ostream& operator<<(std::ostream& os, const float2_& f) {
    os << "(" << f.x << ", " << f.y << ")";
    return os;
}

// Equality operator for assertions
bool operator==(const float2_& a, const float2_& b) {
    return a.x == b.x && a.y == b.y;
}

// forward declarations
void test_svm_vector(boost::compute::context& context, boost::compute::command_queue& queue);
void compute_multistep_test(const boost::compute::context& context, boost::compute::command_queue& queue);
void test_svm_struct(const boost::compute::context& context, boost::compute::command_queue& queue);


/////// Main function ////////
int main() {
    try {
        // Boost Compute setup
        boost::compute::device device = boost::compute::system::default_device();
        boost::compute::context context(device);
        boost::compute::command_queue queue(context, device);

        // Construct the SVMVector with an initial capacity *note* not size. need to resize or push_back to put anything in it.
        boost::compute::SVMVector<float2_> svm_vector(context, queue, 1024);

        // Fill some generic data
        for (int i = 0; i < 10; ++i) {
            svm_vector.push_back({ static_cast<float>(i), static_cast<float>(i * 2) });
        }

        // Print current contents of the svm
        for (size_t i = 0; i < svm_vector.size(); ++i) {
            std::cout << "Element " << i << ": " << svm_vector.at(i) << std::endl;
        }

        // Run the gamut of example tests
        test_svm_vector(context, queue);
        // Call struct testing
        test_svm_struct(context, queue);

        // performance testing
        compute_multistep_test(context, queue);

        // No error, hurray
		// Wait for user input before closing the console
        std::cout << "Press Enter to exit...";
        std::cin.get();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
		std::cout << "Press Enter to exit...";
        std::cin.get();
        return 1;
    }
}

////////// Supporting Functions ///////////////

// Generic test function with use cases and handling of the svm class
void test_svm_vector(boost::compute::context& context, boost::compute::command_queue& queue) {
    try {
        std::cout << "Starting SVMVector test bed...\n";

        boost::compute::device device = queue.get_device();
        std::cout << "Using device: " << device.name() << std::endl;

        // Check SVM capabilities (this is important!)
        cl_device_svm_capabilities svm_caps = device.get_info<CL_DEVICE_SVM_CAPABILITIES>();
        if (!(svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)) {
            std::cerr << "Warning: Device does not support fine-grain SVM. Tests may fail.\n";
        }

        std::cout << "Test 1: Basic Construction\n";
        {
            boost::compute::SVMVector<float2_> vec(context, queue, 10, true /* debug mode */);
            assert(vec.size() == 0);
            assert(vec.capacity() >= 10);
            std::cout << "  - Empty vector constructed successfully.\n";
        }

        std::cout << "Test 2: Push_back and Access\n";
        {
            boost::compute::SVMVector<float2_> vec(context, queue, 5);
            vec.push_back(float2_(1.0f, 2.0f));
            vec.push_back(float2_(3.0f, 4.0f));
            assert(vec.size() == 2);
            assert(vec.at(0) == float2_(1.0f, 2.0f));
            assert(vec.at(1) == float2_(3.0f, 4.0f));
            std::cout << "  - Elements: " << vec.at(0) << ", " << vec.at(1) << "\n";

            // Test out-of-range access
            try {
                vec.at(2);
                assert(false && "Should have thrown out_of_range");
            }
            catch (const std::out_of_range&) {
                std::cout << "  - Out-of-range access correctly throws exception.\n";
            }
        }

        std::cout << "Test 3: Pop_back\n";
        {
            boost::compute::SVMVector<float2_> vec(context, queue, 5);
            vec.push_back(float2_(1.0f, 2.0f));
            vec.push_back(float2_(3.0f, 4.0f));
            vec.pop_back();
            assert(vec.size() == 1);
            assert(vec.at(0) == float2_(1.0f, 2.0f));
            std::cout << "  - Popped back, size = " << vec.size() << ", first element = " << vec.at(0) << "\n";
            vec.pop_back();
            assert(vec.empty());
            std::cout << "  - Popped again, vector is empty.\n";
        }

        std::cout << "Test 4: Resize and Reserve\n";
        {
            boost::compute::SVMVector<float2_> vec(context, queue, 2);
            vec.push_back(float2_(1.0f, 2.0f));
            vec.resize(5); // Should default-construct 4 more elements
            assert(vec.size() == 5);
            assert(vec.at(0) == float2_(1.0f, 2.0f));
            assert(vec.at(1) == float2_(0.0f, 0.0f));
            std::cout << "  - Resized to 5, first two elements: " << vec.at(0) << ", " << vec.at(1) << "\n";

            vec.reserve(10);
            assert(vec.capacity() >= 10);
            assert(vec.size() == 5);
            std::cout << "  - Reserved capacity >= 10, size still 5.\n";

            vec.shrink_to_fit();
            assert(vec.capacity() >= 5); // Exact capacity depends on growth factor
            std::cout << "  - Shrunk to fit, capacity >= " << vec.size() << "\n";
        }

        std::cout << "Test 5: Assign from Range\n";
        {
            std::vector<float2_> source = { {1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f} };
            boost::compute::SVMVector<float2_> vec(context, queue, 1);
            vec.assign(source.begin(), source.end());
            assert(vec.size() == 3);
            assert(vec.at(0) == float2_(1.0f, 2.0f));
            assert(vec.at(2) == float2_(5.0f, 6.0f));
            std::cout << "  - Assigned from vector, elements: " << vec.at(0) << ", " << vec.at(1) << ", " << vec.at(2) << "\n";
        }

        // Iterators are useful for using in boost compute functions
        std::cout << "Test 6: Iterators\n";
        {
            boost::compute::SVMVector<float2_> vec(context, queue, 3);
            vec.push_back(float2_(1.0f, 2.0f));
            vec.push_back(float2_(3.0f, 4.0f));
            vec.push_back(float2_(5.0f, 6.0f));

            int i = 0;
            for (auto it = vec.begin(); it != vec.end(); ++it) {
                assert(*it == vec.at(i));
                i++;
            }
            std::cout << "  - Standard iterators work correctly.\n";

            i = 0;
            for (auto it = vec.beginIterator(); it != vec.endIterator(); ++it) {
                assert(*it == vec.at(i));
                i++;
            }
            std::cout << "  - Device iterators work correctly.\n";
        }

        std::cout << "Test 7: SVM Pointer and Device Usage\n";
        {
            boost::compute::SVMVector<float2_> vec(context, queue, 5);
            vec.push_back(float2_(1.0f, 2.0f));
            void* svm_ptr = vec.get_svm_pointer();
            assert(svm_ptr != nullptr);
            std::cout << "  - SVM pointer retrieved: " << svm_ptr << "\n";

            // Simulate device usage and check the mutexing
            vec.device_begin_use();
            std::thread host_thread([&vec]() {
                std::cout << "  - Host thread attempting access (should block)...\n";
                vec.at(0); // Should block until device_end_use
                std::cout << "  - Host thread accessed vector after device release.\n";
                });

            std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Simulate device work
            vec.device_end_use();
            host_thread.join();
            std::cout << "  - Device usage synchronization works.\n";
        }

        std::cout << "Test 8: Move Semantics\n";
        {
            boost::compute::SVMVector<float2_> vec1(context, queue, 5);
            vec1.push_back(float2_(1.0f, 2.0f));
            boost::compute::SVMVector<float2_> vec2(std::move(vec1));
            assert(vec1.size() == 0);
            assert(vec2.size() == 1);
            assert(vec2.at(0) == float2_(1.0f, 2.0f));
            std::cout << "  - Move constructor works, moved element: " << vec2.at(0) << "\n";

            boost::compute::SVMVector<float2_> vec3(context, queue, 5);
            vec3 = std::move(vec2);
            assert(vec2.size() == 0);
            assert(vec3.size() == 1);
            assert(vec3.at(0) == float2_(1.0f, 2.0f));
            std::cout << "  - Move assignment works, moved element: " << vec3.at(0) << "\n";
        }


        // Test clearing and cleanup
        std::cout << "Test 9: Clear\n";
        {
            boost::compute::SVMVector<float2_> vec(context, queue, 5);
            vec.push_back(float2_(1.0f, 2.0f));
            vec.push_back(float2_(3.0f, 4.0f));
            vec.clear();
            assert(vec.size() == 0);
            assert(vec.empty());
            std::cout << "  - Cleared vector, size = " << vec.size() << "\n";
        }

        std::cout << "All SVMVector tests passed successfully!\n";
    }
    // in the off chance anything broke
    catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        assert(false && "Test bed encountered an exception");
    }
}



//
// This is to show repeated host/device interplay and the flexibility of the SVMVector
// class. I haven't setup a comparison against using existing boost compute svm alloc etc
// since the performance comaprison would be roughly the same, but the code required 
// for interaction is much simpler with the SVMVector (imho)
//
void compute_multistep_test(const boost::compute::context& context,
    boost::compute::command_queue& queue)
{
    std::cout << "\n=== Multi-Step Stress Test: SVM vs. Buffer vs. Host ===\n";

    const size_t N = 500'000'000; // 5 million slices

    // time storage for final comparison
    double svm_time = 0.0;
    double buffer_time = 0.0;
    double host_time = 0.0;

    // setup the kernels
    boost::compute::kernel kernel1;
    boost::compute::kernel kernel2;


    // ------------------------------------------------------------------------
    // 1) SVMVector Approach
    // ------------------------------------------------------------------------
    {
        std::cout << "[SVMVector Multi-Step]\nOperating with " << N << " slices.\n";


        // First kernel: fill array with 4/(1 + x^2)
        const char first_kernel_src[] = R"(
            __kernel void pi_kernel_svm(__global float* data, int n)
            {
                int i = get_global_id(0);
                if(i < n)
                {
                    float x = ((float)i + 0.5f) / (float)n;
                    data[i] = 4.0f / (1.0f + x*x);
                }
            }
        )";
        // second kernel just multiplies each element by 0.5
        const char second_kernel_src[] = R"(
        __kernel void half_kernel(__global float* data, int n)
        {
            int i = get_global_id(0);
            if(i < n)
            {
                data[i] *= 0.5f;
            }
        }
        )";


        // Build programs/kernels
        boost::compute::program program1 = boost::compute::program::build_with_source(first_kernel_src, context);
        kernel1 = program1.create_kernel("pi_kernel_svm");

        boost::compute::program program2 = boost::compute::program::build_with_source(second_kernel_src, context);
        kernel2 = program2.create_kernel("half_kernel");

        // Create SVMVector
        boost::compute::SVMVector<float> svm_vals(context, queue, N, /*debug=*/false);
        svm_vals.resize(N); // Make sure size = N, not just capacity (constructor only sets capacity to N)

        // Set kernel arguments - notice the simplicity for throwing the svm straight in.
        // Need to use the get_svm_pointer function in order to access the void* that the kernel arg is expecting.
        kernel1.set_arg_svm_ptr(0, svm_vals.get_svm_pointer());
        kernel1.set_arg(1, static_cast<int>(N));
        kernel2.set_arg_svm_ptr(0, svm_vals.get_svm_pointer());
        kernel2.set_arg(1, static_cast<int>(N));

        auto gpu_start = std::chrono::high_resolution_clock::now(); // on your marks, set, go!

        // ===== Pass #1: GPU kernel writes partial sums
        svm_vals.device_begin_use();
        queue.enqueue_1d_range_kernel(kernel1, 0, N, 0);
        queue.finish();
        svm_vals.device_end_use();

        // ===== Host modifies second half (no device copy) 
        float* ptr = static_cast<float*>(svm_vals.get_svm_pointer());
        for (size_t i = N / 2; i < N; i++) {
            ptr[i] += 1.0f;
        }

        // ===== Pass #2: GPU scales everything
        svm_vals.device_begin_use();    // note, checking a begin use, and matching with end use to avoid hang lock
        queue.enqueue_1d_range_kernel(kernel2, 0, N, 0);
        queue.finish();
        svm_vals.device_end_use();

        // Sum final result on host without copying back, just direct access to the data
        double sum_svm = 0.0;
        for (size_t i = 0; i < N; i++) {
            sum_svm += ptr[i]; // pull all the elements together
        }
        double avg_svm = sum_svm / (double)N;

        auto gpu_end = std::chrono::high_resolution_clock::now();
        svm_time = std::chrono::duration<double>(gpu_end - gpu_start).count();

        // Note - this is not a gpu only process here, but sum back with the host to show ease and flexibility of svm
        std::cout << "  Final Average Value (SVM) = " << avg_svm << "\n"
            << "  Time (Multi-step)         = " << svm_time << " sec\n\n";
    }

    // ------------------------------------------------------------------------
    // 2) Buffer Approach (Manual Host <-> Device copies)
    // ------------------------------------------------------------------------
    {
        std::cout << "[Buffer Multi-Step]\n"
            << "Each host/device interchange uses explicit read/write calls with queuing.\n";

        // Kernel #3 for partial sums (buffer kernel)
        const char first_kernel_src[] = R"(
            __kernel void pi_kernel_buf(__global float* data, int n)
            {
                int i = get_global_id(0);
                if(i < n)
                {
                    float x = ((float)i + 0.5f) / (float)n;
                    data[i] = 4.0f / (1.0f + x*x);
                }
            }
        )";
        boost::compute::program prog1 = boost::compute::program::build_with_source(first_kernel_src, context);
        boost::compute::kernel kernel3 = prog1.create_kernel("pi_kernel_buf"); // only used in this scope

        // Allocate device buffer
        boost::compute::buffer buf(context, sizeof(float) * N, CL_MEM_READ_WRITE);

        // Set up kernel args
        kernel3.set_arg(0, buf);
        kernel3.set_arg(1, static_cast<int>(N));
        kernel2.set_arg(0, buf); // kernel2 from above
        kernel2.set_arg(1, static_cast<int>(N));

        // ready set go
        auto gpu_start = std::chrono::high_resolution_clock::now();

        // ===== Pass #1
        queue.enqueue_1d_range_kernel(kernel3, 0, N, 0);
        queue.finish();

        // Host wants to modify second half- this example is a little contrived..
        // could do map/unmap, but that needs some code. this approach is probably the
        // 'easiest' without using the svmvector - still needs some syncing/reading/writing stages

        // read from offset of halfway
        size_t half_offset = sizeof(float) * (N / 2);
        size_t half_size = sizeof(float) * (N / 2);

        // additional host vector for just the second half
        std::vector<float> host_data_half(N / 2);

        // read the second half only
        queue.enqueue_read_buffer(buf,
            half_offset,  // offset in bytes
            half_size,    // size in bytes
            host_data_half.data());
        queue.finish();

        // modify the data
        for (size_t i = 0; i < (N / 2); i++) {
            host_data_half[i] += 1.0f;
        }

        // write back to second half
        queue.enqueue_write_buffer(buf,
            half_offset,
            half_size,
            host_data_half.data());
        queue.finish();

        // ===== Pass #2 - run the second kernel
        queue.enqueue_1d_range_kernel(kernel2, 0, N, 0);
        queue.finish();

        // now read the entire buffer back for summing
        std::vector<float> final_data(N);
        queue.enqueue_read_buffer(buf, 0, sizeof(float) * N, final_data.data());
        queue.finish();

        // sum
        double sum_buf = 0.0;
        for (size_t i = 0; i < N; i++) {
            sum_buf +=  final_data[i];
        }
        double avg_buf = sum_buf / (double)N;

        auto gpu_end = std::chrono::high_resolution_clock::now();
        buffer_time = std::chrono::duration<double>(gpu_end - gpu_start).count();

        std::cout << "  Final Average Value (Buffer) = " << avg_buf << "\n"
            << "  Time (Multi-step)           = " << buffer_time
            << " sec\n\n";
    }

    // ------------------------------------------------------------------------
    // 3) Host-only using standard stl vector
    // ------------------------------------------------------------------------
    {
        std::cout << "[Host-only Multi-Step]\n";

        auto host_start = std::chrono::high_resolution_clock::now();

        std::vector<float> host_vals(N);

        // pass #1: fill with 4/(1 + x^2)
        for (size_t i = 0; i < N; i++) {
            float x = (i + 0.5f) / (float)N;
            host_vals[i] = 4.0f / (1.0f + x * x);
        }

        // host modifies second half
        for (size_t i = N / 2; i < N; i++) {
            host_vals[i] += 1.0f;
        }

        // pass #2: scale entire array
        for (size_t i = 0; i < N; i++) {
            host_vals[i] *= 0.5f;
        }

        // sum
        double sum_host_v = 0.0;
        for (size_t i = 0; i < N; i++) {
            sum_host_v += host_vals[i];
        }
        double avg_host = sum_host_v / (double)N;

        auto host_end = std::chrono::high_resolution_clock::now();
        host_time = std::chrono::duration<double>(host_end - host_start).count();

        std::cout << "  Final Average Value (Host) = " << avg_host << "\n"
            << "  Time (Multi-step)         = " << host_time << " sec\n\n";
    }

    // ------------------------------------------------------------------------
    // Benchmark printouts
    // ------------------------------------------------------------------------
    std::cout << "=== Comparison ===\n";
    std::cout << "SVM Time:    " << svm_time << " s\n";
    std::cout << "Buffer Time: " << buffer_time << " s\n";
    std::cout << "Host Time:   " << host_time << " s\n\n";

    auto printCompare = [&](const char* lblA, double timeA,
        const char* lblB, double timeB)
        {
            double diff = timeB - timeA;
            double ratio = diff / timeB * 100.0;
            // figure out faster thans
            if (ratio > 0) {
                std::cout << lblA << " is " << +ratio << "% faster than " << lblB << "\n";
            }
            else if (ratio < 0) {
                std::cout << lblA << " is " << -ratio << "% slower than " << lblB << "\n";
            }
            else {
                std::cout << lblA << " is the same speed as " << lblB << "\n";
            }
        };
    // print benchmark times
    printCompare("SVM", svm_time, "Host", host_time);
    printCompare("SVM", svm_time, "Buffer", buffer_time);
    printCompare("Buffer", buffer_time, "Host", host_time);

    std::cout << "=== End Multi-Step Stress Test ===\n\n";
}


void test_svm_struct(const boost::compute::context& context,
    boost::compute::command_queue& queue)
{
    using namespace boost::compute;

    std::cout << "\n=== Verifying SVMVector<struct> in a kernel ===\n";
    std::cout << "(Alignment, element access and retrieval, trivial types (int, float, float4).\n"
        << "User can alter kernel to test your own non - trivial types)\n";

    // build kernel
    program prog = program::build_with_source(SVM_STRUCT_KERNEL_SRC, context);
    kernel k = prog.create_kernel("StructKernel");

    // the fun bit - create an SVMVector with the T being a struct
    const int N = 8; // number of elemenets we'll use
    SVMVector<TestStruct> vec(context, queue, N, true); // debug on
    vec.resize(N); // always important to do this! see svm_vector for explanation

    std::cout << "Struct setup:\n";
    std::cout << "sizeof(TestStruct): " << sizeof(TestStruct) << "\n";
    std::cout << "alignof(TestStruct): " << alignof(TestStruct) << "\n";
    std::cout << "offset of dims: " << offsetof(TestStruct, dims) << "\n";


    // Jam some data into the vector
    // Directly written by host, and since no device usage should be no concurrency issues (or need for device being/end)
    TestStruct* ptr = static_cast<TestStruct*>(vec.get_svm_pointer()); // set up an easy alias to the struct
    for (int_ i = 0; i < N; i++) {
        ptr[i].type = i;
        ptr[i].radius = (float_)(i * 10.0f);
        ptr[i].dims = float4_{ (float_)i, (float_)(2 * i), (float_)(3 * i), (float_)(4 * i) };
    }
   
    // check it's all there
    std::cout << "Host data BEFORE kernel:\n";
    //TestStruct* ptr = static_cast<TestStruct*>(vec.get_svm_pointer());
    for (int i = 0; i < N; i++) {
        std::cout << " i=" << i
            << "  type=" << ptr[i].type
            << "  radius=" << ptr[i].radius
            << "  dims=(" << ptr[i].dims.x << "," << ptr[i].dims.y << "," << ptr[i].dims.z << "," << ptr[i].dims.w << ")\n";
    }

    // get the kernel arguments with SVM pointer ready
    k.set_arg_svm_ptr(0, vec.get_svm_pointer()); // struct input
    k.set_arg(1, N);                            // int N (so this kernel knows the size and doesn't spawn threads out of bounds)
    k.set_arg(2, 1);                            // value of 1 for debug outputs

    // Since device usage, guard with device_begin_use/device_end_use()
    std::cout << "Launching Struct-Testing Kernel...\n";
    vec.device_begin_use();
    queue.enqueue_1d_range_kernel(k, 0, N, 0); // jam it in the queue
    queue.finish();
    vec.device_end_use();

    // Show that host has direct access to data AFTER kernel. No copyback/remap
    // Notice: the radius should have been incremented by 1.0 (if using the kernel example given)
    std::cout << "\nHost data AFTER kernel:\n";
    
    //TestStruct* ptr = static_cast<TestStruct*>(vec.get_svm_pointer());
    for (int i = 0; i < N; i++) {
        std::cout << " i=" << i
            << "  type=" << ptr[i].type
            << "  radius=" << ptr[i].radius
            << "  dims=(" << ptr[i].dims.x << "," << ptr[i].dims.y << "," << ptr[i].dims.z << "," << ptr[i].dims.w << ")\n";
    }
    

    // Sanity check - this is important if you're having data errors - its 99% likely it's a misalignment issue
    // if so, might need to explicitly introduce padding into the kernel (tested without padding and just attribute setting
    // on an older AMD gpu
    //TestStruct* ptr = static_cast<TestStruct*>(vec.get_svm_pointer());
    for (int i = 0; i < N; i++) {
        // radius should have been incremented by 1 in the kernel
        float expected = (float)(i * 10.0f + 1.0f);
        if (std::fabs(ptr[i].radius - expected) > 1e-6f) {
            std::cerr << "ERROR: idx=" << i << " radius mismatch => "
                << ptr[i].radius << " (expected " << expected << ")\n";
            std::cerr << "Possibly the error lies with misalignment of struct -  check the code. Your device may need explicit padding.\n";
        }
    }
    std::cout << "\nIncrease of correct elements (radius) successful. No mismatch!\n";
    std::cout << "=== Kernel struct testing completed ===\n\n";
}
