cmake_minimum_required(VERSION 3.18)
project(SVMVectorBoostExample)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Single cached variable for the user to specify where boost/compute.hpp is located:
set(BOOST_COMPUTE_ROOT_DIR "" CACHE PATH "Path to Boost.Compute directory (must contain boost/compute.hpp)")

# If not specified, fail:
if(NOT BOOST_COMPUTE_ROOT_DIR)
    message(FATAL_ERROR "Please set BOOST_COMPUTE_ROOT_DIR to the boot root directory")
endif()

# Directly check that the file exists, no find_path or caching:
if(NOT EXISTS "${BOOST_COMPUTE_ROOT_DIR}/boost/compute.hpp")
    message(FATAL_ERROR "Could not find 'boost/compute.hpp' under: ${BOOST_COMPUTE_ROOT_DIR}")
endif()

# Find OpenCL
find_package(OpenCL REQUIRED)

add_executable(SVMVector_for_Boost
    SVMVector_Examples.cpp
    svm_vector.hpp
    struct_test.h
)

# Include directories
target_include_directories(SVMVector_for_Boost PRIVATE
    "${BOOST_COMPUTE_ROOT_DIR}"
    "${OpenCL_INCLUDE_DIRS}"
)

# On Windows, link to opencl.lib if found. Otherwise fallback to dynamic loading.
if(WIN32)
    if(OpenCL_LIBRARIES)
        message(STATUS "Linking to OpenCL: ${OpenCL_LIBRARIES}")
        target_link_libraries(SVMVector_for_Boost PRIVATE ${OpenCL_LIBRARIES})
    else()
        message(WARNING "No OpenCL.lib found") # Boost doesn't readily support dynamic opencl
    endif()
else()
    # Non-Windows -> link normally
    target_link_libraries(SVMVector_for_Boost PRIVATE ${OpenCL_LIBRARIES})
endif()

# Generate compile_commands.json for IDE tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Organise files in VisualStudio filters
source_group("Header Files" FILES svm_vector.hpp)
source_group("Source Files" FILES SVMVector_Examples.cpp)

# Print some status
message(STATUS "Boost.Compute include dir: ${BOOST_COMPUTE_ROOT_DIR}")
message(STATUS "OpenCL include dirs: ${OpenCL_INCLUDE_DIRS}")
message(STATUS "OpenCL libraries: ${OpenCL_LIBRARIES}")
