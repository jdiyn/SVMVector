@echo off
setlocal

REM Optional override paths
set "BOOST_COMPUTE_ROOT_DIR=C:\Packages\boost_1_86_0"
set "OPENCL_ROOT="

REM Create build folder if it doesn't exist
if not exist build (
    mkdir build
)
cd build

REM Run CMake to configure the project
cmake .. ^
  -G "Visual Studio 17 2022" -A x64 ^
  -DBOOST_COMPUTE_ROOT_DIR="%BOOST_COMPUTE_ROOT_DIR%" ^
  -DOPENCL_ROOT="%OPENCL_ROOT%"

REM Build the project in Release configuration
cmake --build . --config Release

cd ..
endlocal
pause
