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
//
// SVMVector.hpp - Shared Virtual Memory Vector for Boost.Compute OpenCL
// Author: Joshua Diyn
// Copyright (c) 2024
// Date: 17/10/2024
//
// This file is a simpler way to interact with Boost's implementation of the
// Shared Virtual Memory (SVM). It provides a vector-like container that uses
// fine-grained SVM in OpenCL 2.0 or later. It allows both host and device
// to access the same underlying memory without copying data back and forth.
// A simple locking mechanism is included to avoid race conditions:
//   - device_begin_use() indicates the device is about to operate on the vector,
//   - device_end_use() indicates completion,
// so host operations block if the device is currently using the vector.
// 
// Dependencies:
// 1. Ensure your project has boost compute headers as additional includes
// 2. Ensure project includes cl directory
// 3. Ensure opencl.lib included in project
//
// Usage example:
//
//   // construction
//   boost::compute::SVMVector<float2_> svm_vector(context, queue, initial_capacity);
//
//   // fill, push/pop, etc.
//   for (int i = 0; i < 1000; i++) {
//       svm_vector.push_back({(float)i, (float)(i * 2)});
//   }
//
//   // pass to OpenCL kernel
//   kernel.set_arg_svm_ptr(0, svm_vector.get_svm_pointer());
//   svm_vector.device_begin_use();
//   queue.enqueue_1d_range_kernel(kernel, 0, svm_vector.size(), 0);
//   queue.finish();
//   svm_vector.device_end_use();
//
// Methods provided include: push_back, pop_back, at, set, resize, reserve,
// shrink_to_fit, assign, clear, begin/end iterators, and get_svm_pointer().
// Copy constructors are disabled; only move semantics are supported.
// Allocations are performed with boost::compute::svm_alloc / svm_free and
// automatically expand as needed.
//
// Ensure your device supports CL_DEVICE_SVM_FINE_GRAIN_BUFFER, or else this
// class will not function properly.
// 
// 
// TODO::
//  - for future use, decouple this from boost compute entirely & just rely on cl
//  - a CL_DEVICE_SVM_FINE_GRAIN_BUFFER check in the constructor to ensure support
// 
// ============================================================================


/*
// Example main() demonstrating use of SVMVector:

#include <iostream>
#include <boost/compute.hpp>
#include "svm_vector.hpp"

// Simple structure for testing, e.g. float2_
struct float2_ {
    float x, y;
};

// For printing float2_ to std::ostream
std::ostream& operator<<(std::ostream &os, const float2_ &f) {
    os << "(" << f.x << ", " << f.y << ")";
    return os;
}

int main()
{
    try {
        // Select a device and create a context + queue
        boost::compute::device device = boost::compute::system::default_device();
        boost::compute::context context(device);
        boost::compute::command_queue queue(context, device);

        // Construct the SVMVector with an initial capacity
        boost::compute::SVMVector<float2_> svm_vector(context, queue, 1024);

        // Fill some data
        for(int i = 0; i < 10; ++i) {
            svm_vector.push_back({(float)i, (float)(i * 2)});
        }

        // Print current contents
        for(size_t i = 0; i < svm_vector.size(); ++i) {
            std::cout << "Element " << i << ": " << svm_vector.at(i) << std::endl;
        }

        // Done
        return 0;
    }
    catch(const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
*/


#pragma once

#include <boost/compute/svm.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/wait_list.hpp>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <mutex>
#include <memory>
#include <condition_variable>
#include <atomic>
#include <cmath>        
#include <type_traits>

namespace boost {
namespace compute {


// svm_device_iterator first, so we can make use of the iterative parts in boost
template <typename T>
struct svm_device_iterator {
    typedef std::ptrdiff_t difference_type;
    typedef T value_type;
    typedef T* pointer;
    typedef T& reference;
    typedef std::random_access_iterator_tag iterator_category;

    svm_device_iterator() : ptr(nullptr) {}
    explicit svm_device_iterator(T* p) : ptr(p) {}

    reference operator*() const { return *ptr; }
    pointer operator->() const { return ptr; }
    svm_device_iterator& operator++() {
        ++ptr;
        return *this;
    }
    svm_device_iterator operator++(int) {
        svm_device_iterator tmp(*this);
        ++ptr;
        return tmp;
    }
    svm_device_iterator& operator--() {
        --ptr;
        return *this;
    }
    svm_device_iterator operator--(int) {
        svm_device_iterator tmp(*this);
        --ptr;
        return tmp;
    }
    svm_device_iterator& operator+=(difference_type n) {
        ptr += n;
        return *this;
    }
    svm_device_iterator operator+(difference_type n) const { return svm_device_iterator(ptr + n); }
    svm_device_iterator& operator-=(difference_type n) {
        ptr -= n;
        return *this;
    }
    svm_device_iterator operator-(difference_type n) const { return svm_device_iterator(ptr - n); }
    difference_type operator-(const svm_device_iterator& other) const { return ptr - other.ptr; }
    bool operator==(const svm_device_iterator& other) const { return ptr == other.ptr; }
    bool operator!=(const svm_device_iterator& other) const { return ptr != other.ptr; }
    bool operator<(const svm_device_iterator& other) const { return ptr < other.ptr; }
    bool operator>(const svm_device_iterator& other) const { return ptr > other.ptr; }
    bool operator<=(const svm_device_iterator& other) const { return ptr <= other.ptr; }
    bool operator>=(const svm_device_iterator& other) const { return ptr >= other.ptr; }
    reference operator[](difference_type n) const { return ptr[n]; }    // Important! This does no bounds checking!

    T* ptr;
};

// Specialised is_device_iterator
template <typename T>
struct is_device_iterator<svm_device_iterator<T>> : boost::true_type {};


//////////////////////////////////////////// Main Class ////////////////////////////////////////////////////////////
template <typename T>
class SVMVector {
    // check if trivially copyable.. could remove this, but would require some code to handle non-trivial types well
    static_assert(std::is_trivially_copyable<T>::value, "SVMVector requires T to be trivially copyable.");

  public:
    // Constructor
    SVMVector(const compute::context& context,
              compute::command_queue& queue,
              size_t initial_capacity = 1024,
              bool debug_mode = false,
              float growth_factor = 1.5f)
        : m_context(std::make_shared<const compute::context>(context)),
          m_queue(std::make_shared<compute::command_queue>(queue)),
          m_size(0),
          m_capacity(initial_capacity > 0 ? initial_capacity : 16),      // basic starting size
          m_device_in_use(false),
          m_debug(debug_mode),
          m_growth_factor(growth_factor < 1.0f ? 1.0f : growth_factor) { // need to ensure growth is at least a factor of 1


        allocate_svm_memory(m_capacity);
        if (m_debug) {
            std::cout << "[DEBUG] SVMVector constructed with initial capacity: " << m_capacity.load() << std::endl;
        }
    }

    // Destructor
    ~SVMVector() {
        // Before freeing memory, destroy any remaining objects
        size_t sz = m_size.load(std::memory_order_relaxed);
        destroy_range(0, sz);

        free_svm_memory();
        if (m_debug) {
            std::cout << "[DEBUG] SVMVector destructed." << std::endl;
        }
    }

    // Delete copy constructor and copy assignment
    SVMVector(const SVMVector&) = delete;
    SVMVector& operator=(const SVMVector&) = delete;

    // Move constructor
    SVMVector(SVMVector&& other) noexcept
        : m_data(std::move(other.m_data)),
          m_context(std::move(other.m_context)),
          m_queue(std::move(other.m_queue)),
          m_size(other.m_size.load(std::memory_order_relaxed)),
          m_capacity(other.m_capacity.load(std::memory_order_relaxed)),
          m_device_in_use(other.m_device_in_use.load(std::memory_order_relaxed)),
          m_debug(other.m_debug),
          m_growth_factor(other.m_growth_factor) {
        other.m_size.store(0, std::memory_order_relaxed);
        other.m_capacity.store(0, std::memory_order_relaxed);
        other.m_device_in_use.store(false, std::memory_order_relaxed);

        if (m_debug) {
            std::cout << "[DEBUG] SVMVector move constructed." << std::endl;
        }
    }

    // Move assignment
    SVMVector& operator=(SVMVector&& other) noexcept {
        if (this != &other) {
            // Destroy existing elements
            size_t sz = m_size.load(std::memory_order_relaxed);
            destroy_range(0, sz);
            free_svm_memory();

            m_data = std::move(other.m_data);
            m_context = std::move(other.m_context);
            m_queue = std::move(other.m_queue);
            m_size.store(other.m_size.load(std::memory_order_relaxed), std::memory_order_relaxed);
            m_capacity.store(other.m_capacity.load(std::memory_order_relaxed), std::memory_order_relaxed);
            m_device_in_use.store(other.m_device_in_use.load(std::memory_order_relaxed), std::memory_order_relaxed);
            m_debug = other.m_debug;
            m_growth_factor = other.m_growth_factor;

            other.m_size.store(0, std::memory_order_relaxed);
            other.m_capacity.store(0, std::memory_order_relaxed);
            other.m_device_in_use.store(false, std::memory_order_relaxed);

            if (m_debug) {
                std::cout << "[DEBUG] SVMVector move assigned." << std::endl;
            }
        }
        return *this;
    }

    // Indicate that device is about to use the vector
    void device_begin_use() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_condition_variable.wait(lock, [this] { return !m_device_in_use.load(std::memory_order_acquire); });
        m_device_in_use.store(true, std::memory_order_release);
        if (m_debug) {
            std::cout << "[DEBUG] Device has begun using the SVMVector." << std::endl;
        }
    }

    // Indicate that device has finished using the vector
    void device_end_use() {
        // Ensure queued operations have completed
        m_queue->finish();
        // handle lock guard
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_device_in_use.store(false, std::memory_order_release);
            if (m_debug) {
                std::cout << "[DEBUG] Device has finished using the SVMVector." << std::endl;
            }
        }
        m_condition_variable.notify_all();
    }

    // Is vector empty?
    bool empty() const { return (m_size.load(std::memory_order_relaxed) == 0); }

    // vector like construct a new element at the end
    void push_back(const T& value) {
        std::unique_lock<std::mutex> lock(m_mutex);
        wait_until_device_not_in_use(lock);

        size_t curr_size = m_size.load(std::memory_order_relaxed);
        size_t curr_capacity = m_capacity.load(std::memory_order_relaxed);
        // error check so as not to overflow
        if (curr_capacity > (std::numeric_limits<size_t>::max() / m_growth_factor)) {
            throw std::runtime_error("Capacity overflow in push_back");
        }

        if (curr_size >= curr_capacity) {
            size_t new_capacity = static_cast<size_t>(std::ceil(curr_capacity * m_growth_factor));
            reserve_internal(new_capacity);
            curr_capacity = m_capacity.load(std::memory_order_relaxed);
        }

        // Placement-new the object in place
        new (&(get_data_ptr()[curr_size])) T(value);

        m_size.store(curr_size + 1, std::memory_order_relaxed);

        if (m_debug) {
            std::cout << "[DEBUG] push_back: new size = " << (curr_size + 1) << std::endl;
        }
    }

    // vector like pop the last element
    void pop_back() {
        std::unique_lock<std::mutex> lock(m_mutex);
        wait_until_device_not_in_use(lock);

        size_t curr_size = m_size.load(std::memory_order_relaxed);
        if (curr_size > 0) {
            curr_size--;
            m_size.store(curr_size, std::memory_order_relaxed);

            // Call destructor for the removed element
            if constexpr (!std::is_trivially_destructible<T>::value) {
                get_data_ptr()[curr_size].~T();
            }
            if (m_debug) {
                std::cout << "[DEBUG] pop_back: new size = " << curr_size << std::endl;
            }
        }
    }

    // at() => with bounds checking
    // note: this locks internally to ensure the device isn’t in use. This can result in significant o/head or slow down
    // if safely done, a faster way to access is to ensure the device is not in use, and instead access raw pointers.
    T& at(size_t index) {
        std::unique_lock<std::mutex> lock(m_mutex);
        wait_until_device_not_in_use(lock);

        if (index >= m_size.load(std::memory_order_relaxed)) {
            throw std::out_of_range("Index out of range: " + std::to_string(index) +
                ". size() = " + std::to_string(m_size.load(std::memory_order_relaxed)) +
                "\nDid you set capacity but forget to size/resize/push_back?");
        }
        return get_data_ptr()[index];
    }

    const T& at(size_t index) const {
        std::unique_lock<std::mutex> lock(m_mutex);
        wait_until_device_not_in_use(lock);

        if (index >= m_size.load(std::memory_order_relaxed)) {
            throw std::out_of_range("Index out of range: " + std::to_string(index) +
                ". size() = " + std::to_string(m_size.load(std::memory_order_relaxed)) +
                    "\n Did you set capacity but forget to size/resize/push_back?");

        }
        return get_data_ptr()[index];
    }

    // fast access, but need to be careful and handle well in client code to avoid race conditions and/or
    // out of bounds access/undefined behaviour
    T& operator[](size_t index) {
        // No locking, no device check, no range check
        return get_data_ptr()[index];
    }

    const T& operator[](size_t index) const {
        // likewise, no checks
        return get_data_ptr()[index];
    }


    // no read, just store
    void set(size_t index, const T& value) {
        std::unique_lock<std::mutex> lock(m_mutex);
        wait_until_device_not_in_use(lock);

        if (index >= m_size.load(std::memory_order_relaxed)) {
            throw std::out_of_range("Index out of range");
        }
        get_data_ptr()[index] = value;
    }

    // Expose raw SVM pointer
    void* get_svm_pointer() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_data.get()) {
            throw std::runtime_error("SVM pointer is null");
        }
        return m_data.get();
    }

    // size + capacity
    size_t size() const { return m_size.load(std::memory_order_relaxed); }
    size_t capacity() const { return m_capacity.load(std::memory_order_relaxed); }

    // clear() => destroy all elements, size=0
    void clear() {
        std::unique_lock<std::mutex> lock(m_mutex);
        wait_until_device_not_in_use(lock);

        size_t sz = m_size.load(std::memory_order_relaxed);
        destroy_range(0, sz);
        m_size.store(0, std::memory_order_relaxed);

        if (m_debug) {
            std::cout << "[DEBUG] clear: size reset to 0" << std::endl;
        }
    }

    // resize => if new_size>old => default-construct new elements
    //           if new_size<old => destruct old elements
    void resize(size_t new_size) {
        std::unique_lock<std::mutex> lock(m_mutex);
        wait_until_device_not_in_use(lock); // if hanging here, there might be a failure to pair a device_begin_us with device_end_use somewhere in the application

        size_t curr_size = m_size.load(std::memory_order_relaxed);
        size_t curr_capacity = m_capacity.load(std::memory_order_relaxed);

        if (new_size > curr_capacity) {
            reserve_internal(new_size);
            curr_capacity = m_capacity.load(std::memory_order_relaxed);
        }

        T* data_ptr = get_data_ptr();
        if (new_size > curr_size) {
            // Construct new objects [curr_size..new_size)
            for (size_t i = curr_size; i < new_size; i++) {
                new (&data_ptr[i]) T();
            }
        } else if (new_size < curr_size) {
            // Destroy [new_size..curr_size)
            destroy_range(new_size, curr_size);
        }
        m_size.store(new_size, std::memory_order_relaxed);

        if (m_debug) {
            std::cout << "[DEBUG] resize: new size = " << new_size << std::endl;
        }
    }

    // reserve => ensure capacity is at least new_capacity
    void reserve(size_t new_capacity) {
        std::unique_lock<std::mutex> lock(m_mutex);
        wait_until_device_not_in_use(lock);
        reserve_internal(new_capacity);
    }

    // shrink_to_fit => reduce capacity to current size
    void shrink_to_fit() {
        std::unique_lock<std::mutex> lock(m_mutex);
        wait_until_device_not_in_use(lock);

        size_t sz = m_size.load(std::memory_order_relaxed);
        if (m_capacity.load(std::memory_order_relaxed) > sz) {
            reserve_internal(sz);
        }
    }

    // assign from a range
    template <typename InputIterator>
    void assign(InputIterator first, InputIterator last) {
        using ValueType = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_same<ValueType, T>::value, "Input type must match SVMVector value type");

        std::unique_lock<std::mutex> lock(m_mutex);
        wait_until_device_not_in_use(lock);

        size_t new_size = static_cast<size_t>(std::distance(first, last));
        if (new_size > m_capacity.load(std::memory_order_relaxed)) {
            reserve_internal(new_size);
        }

        // Destroy the old elements if new_size < current_size
        size_t curr_size = m_size.load(std::memory_order_relaxed);
        if (new_size < curr_size) {
            destroy_range(new_size, curr_size);
        }

        T* data_ptr = get_data_ptr();
        size_t idx = 0;
        for (auto it = first; it != last; ++it) {
            if (idx < curr_size) {
                data_ptr[idx] = *it;
            } else {
                // Need to placement-new
                new (&data_ptr[idx]) T(*it);
            }
            idx++;
        }

        // If new_size>curr_size, we've constructed new_size-curr_size elements
        // If new_size<curr_size, we've destroyed the difference above

        m_size.store(new_size, std::memory_order_relaxed);
        if (m_debug) {
            std::cout << "[DEBUG] assign: new size = " << new_size << std::endl;
        }
    }
    // updates the vector’s data without locking its own mutex. This assumes the caller has already ensured exclusive access
    template <typename InputIterator, typename Converter>
    void assign_from_no_lock(InputIterator first, InputIterator last, Converter converter) {
        size_t new_size = std::distance(first, last);
        if (new_size > m_capacity) {
            reserve_internal(new_size);
        }
        T* data_ptr = get_data_ptr();
        size_t idx = 0;
        for (auto it = first; it != last; ++it) {
            data_ptr[idx] = converter(*it);
            idx++;
        }
        m_size = new_size;
    }

    // overload to allow compatible imports assign
    template <typename InputIterator>
    void assign_from_no_lock(InputIterator first, InputIterator last) {
        using SourceType = typename std::iterator_traits<InputIterator>::value_type;
        assign_from_no_lock(first, last, [](const SourceType& val) { return static_cast<T>(val); });
    }

    // conversion assign from
    template <typename InputIterator, typename Converter>
    void assign_from(InputIterator first, InputIterator last, Converter converter) {
        using SourceType = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_same<decltype(converter(std::declval<SourceType>())), T>::value,
                      "Converter must produce type T");

        std::unique_lock<std::mutex> lock(m_mutex);
        wait_until_device_not_in_use(lock);

        size_t new_size = static_cast<size_t>(std::distance(first, last));
        if (new_size > m_capacity.load(std::memory_order_relaxed)) {
            reserve_internal(new_size);
        }

        // Destroy old elements if new_size < current_size
        size_t curr_size = m_size.load(std::memory_order_relaxed);
        if (new_size < curr_size) {
            destroy_range(new_size, curr_size);
        }

        T* data_ptr = get_data_ptr();
        size_t idx = 0;
        for (auto it = first; it != last; ++it) {
            T converted = converter(*it);
            if (idx < curr_size) {
                data_ptr[idx] = std::move(converted);  // Overwrite existing element
            } else {
                new (&data_ptr[idx]) T(std::move(converted));  // Construct new element
            }
            idx++;
        }

        m_size.store(new_size, std::memory_order_relaxed);

        if (m_debug) {
            std::cout << "[DEBUG] assign_from: new size = " << new_size << std::endl;
        }
    }

    // Iterators
    T* begin() {
        std::unique_lock<std::mutex> lock(m_mutex);
        wait_until_device_not_in_use(lock);
        return get_data_ptr();
    }
    T* end() {
        std::unique_lock<std::mutex> lock(m_mutex);
        wait_until_device_not_in_use(lock);
        return get_data_ptr() + m_size.load(std::memory_order_relaxed);
    }
    const T* begin() const {
        std::unique_lock<std::mutex> lock(m_mutex);
        wait_until_device_not_in_use(lock);
        return get_data_ptr();
    }
    const T* end() const {
        std::unique_lock<std::mutex> lock(m_mutex);
        wait_until_device_not_in_use(lock);
        return get_data_ptr() + m_size.load(std::memory_order_relaxed);
    }

    // flush_host_writes => finish the queue
    void flush_host_writes() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue->finish();
        if (m_debug) {
            std::cout << "[DEBUG] flush_host_writes: queue finished." << std::endl;
        }
    }

    // set growth factor (perhaps if the constructor growth factor needs adjustment
    void set_growth_factor(float factor) {
        if (factor < 1.0f) {
            factor = 1.0f;
        }
        m_growth_factor = factor;
    }

    //
    // Iterators:
    svm_device_iterator<T> beginIterator() {
        std::unique_lock<std::mutex> lock(m_mutex);
        wait_until_device_not_in_use(lock);
        return svm_device_iterator<T>(get_data_ptr());
    }

    svm_device_iterator<T> endIterator() {
        std::unique_lock<std::mutex> lock(m_mutex);
        wait_until_device_not_in_use(lock);
        return svm_device_iterator<T>(get_data_ptr() + m_size.load(std::memory_order_relaxed));
    }

    svm_device_iterator<const T> beginIterator() const {
        std::unique_lock<std::mutex> lock(m_mutex);
        wait_until_device_not_in_use(lock);
        return svm_device_iterator<const T>(get_data_ptr());
    }

    svm_device_iterator<const T> endIterator() const {
        std::unique_lock<std::mutex> lock(m_mutex);
        wait_until_device_not_in_use(lock);
        return svm_device_iterator<const T>(get_data_ptr() + m_size.load(std::memory_order_relaxed));
    }

  private:
    compute::svm_ptr<T> m_data; // boost compute reliant svm_ptr for storage of the data. Any T must be compatible

    std::shared_ptr<const compute::context> m_context;
    std::shared_ptr<compute::command_queue> m_queue;

    std::atomic<size_t> m_size;
    std::atomic<size_t> m_capacity;
    // concurrency checks
    mutable std::mutex m_mutex;
    std::condition_variable m_condition_variable;
    std::atomic<bool> m_device_in_use;

    bool m_debug;   // enables all console printouts
    float m_growth_factor;

    // Some private helpers
    // 
    void wait_until_device_not_in_use(std::unique_lock<std::mutex>& lock) {
        m_condition_variable.wait(lock, [this] { return !m_device_in_use.load(std::memory_order_acquire); });
    }

    // get raw data ptr (non-const)
    T* get_data_ptr() {
        if (!m_data.get()) {
            throw std::runtime_error("SVM pointer is null");
        }
        return static_cast<T*>(m_data.get());
    }

    // Get raw data ptr (const)
    const T* get_data_ptr() const {
        if (!m_data.get()) {
            throw std::runtime_error("SVM pointer is null");
        }
        return static_cast<const T*>(m_data.get());
    }

    // allocate SVM memory
    void allocate_svm_memory(size_t capacity) {
        if (capacity == 0) {
            m_data = compute::svm_ptr<T>();
            return;
        }
        m_data = compute::svm_alloc<T>(*m_context, capacity, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER);
        if (!m_data.get()) {
            throw std::runtime_error("Failed to allocate SVM memory");
        }
    }

    // free up the svm
    void free_svm_memory()
    {
        // Only free if we still have a non-null context and a non-null SVM pointer
        if (m_context && m_data.get()) {
            boost::compute::svm_free(*m_context, m_data);
        }
        // Reset m_data to a null state. That way, double frees are avoided
        m_data = boost::compute::svm_ptr<T>();
    }

        
    // destroy elements in [first, last)
    void destroy_range(size_t first, size_t last) {
        if constexpr (!std::is_trivially_destructible<T>::value) {
            T* ptr = get_data_ptr();
            for (size_t i = first; i < last; i++) {
                ptr[i].~T();
            }
        }
    }

    // reserve internal space
    void reserve_internal(size_t new_capacity) {
        size_t old_capacity = m_capacity.load(std::memory_order_relaxed);
        if (new_capacity <= old_capacity)
            return;

        if (m_debug) {
            std::cout << "[DEBUG] reserve_internal: old_capacity = " << old_capacity
                      << ", new_capacity = " << new_capacity << std::endl;
        }

        compute::svm_ptr<T> new_data;
        try {
            new_data =
                compute::svm_alloc<T>(*m_context, new_capacity, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER);  // if the device istn' fine grain supportive, this entire class will fail.
        } catch (...) {
            if (m_debug) {
                std::cerr << "[DEBUG] Failed to allocate new SVM memory in reserve_internal." << std::endl;
            }
            throw;
        }

        if (!new_data.get()) {
            if (m_debug)
                std::cerr << "[DEBUG] Alloc returned null. Trying fallback." << std::endl;

            // fallback approach
            size_t fallback_capacity = (old_capacity + new_capacity) / 2; // halfway attempt to be graceful before failure
            if (fallback_capacity > old_capacity) {
                try {
                    new_data = compute::svm_alloc<T>(*m_context, fallback_capacity,
                                                     CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER);
                } catch (...) {
                    if (m_debug) {
                        std::cerr << "[DEBUG] Fallback also failed." << std::endl;
                    }
                    throw;
                }
                if (new_data.get()) {
                    new_capacity = fallback_capacity;
                } else {
                    throw std::runtime_error("Failed to allocate fallback in reserve_internal()");
                }
            } else {
                throw std::runtime_error("Failed to expand capacity");
            }
        }

        size_t curr_size = m_size.load(std::memory_order_relaxed);

        T* old_ptr = static_cast<T*>(m_data.get());
        T* new_ptr = static_cast<T*>(new_data.get());


        // If T is trivially copyable then memcpy (possible exapnsion here if using non trivial,
        // but there's a compile time check at the beginnign to restrict T. Worth keeping for future use
        if constexpr (std::is_trivially_copyable<T>::value) {
            std::memcpy(new_ptr, old_ptr, curr_size * sizeof(T));
        } else {
            // do a move-construct each element, then destroy old
            for (size_t i = 0; i < curr_size; i++) {
                new (&new_ptr[i]) T(std::move(old_ptr[i]));
                old_ptr[i].~T();
            }
        }

        // free old memory
        compute::svm_free(*m_context, m_data);

        m_data = new_data;
        m_capacity.store(new_capacity, std::memory_order_relaxed);

        if (m_debug) {
            std::cout << "[DEBUG] reserve_internal: new capacity = " << new_capacity << std::endl;
        }
    }

}; // end of class
}  // namespace compute
}  // namespace boost
