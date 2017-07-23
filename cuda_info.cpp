#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

enum class CudaArch
{
    Unknown,
    Fermi = 2,
    Kepler = 3,
    Maxwell = 5,
    Pascal = 6
};


std::ostream& operator<<(std::ostream& os, CudaArch const& a)
{
    switch(a)
    {
        case CudaArch::Fermi:
            os << "Fermi";
            break;
        case CudaArch::Kepler:
            os << "Kepler";
            break;
        case CudaArch::Maxwell:
            os << "Maxwell";
            break;
        case CudaArch::Pascal:
            os << "Pascal";
            break;
    }
    return os;
}

CudaArch getArch(cudaDeviceProp const& prop)
{
    CudaArch arch;
    switch(prop.major)
    {
        case 2: arch = CudaArch::Fermi;    break;
        case 3: arch = CudaArch::Kepler;   break;
        case 5: arch = CudaArch::Maxwell;  break;
        case 6: arch = CudaArch::Pascal;   break;
        default: arch = CudaArch::Unknown; break;
    }
    return arch;
}

int getCores(cudaDeviceProp const& prop)
{
    auto mp = prop.multiProcessorCount;
    switch(getArch(prop))
    {
        case CudaArch::Fermi:
            return mp * (prop.minor == 1 ? 32 : 48);
        case CudaArch::Kepler:
            return mp * 192;
        case CudaArch::Maxwell:
            return mp * 128;
        case CudaArch::Pascal:
            return mp * (prop.minor == 0 ? 64 : 128);
        default:
            std::cerr << "Unknown device type." << std::endl;
    }
    return 0;
}

auto main(int argc, char ** argv) -> int
{
    int deviceCount;

    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) 
    {
        cudaDeviceProp deviceProp;

        auto error = cudaGetDeviceProperties(&deviceProp, dev);
        if(error != cudaSuccess)
        {
            std::cerr << "Failed to get device properties for device #"
                      << dev
                      << std::endl;
            return -1;
        }

        std::cout << "Device #" << dev << " name: " << deviceProp.name 
                  << "\n Computational Capabilities: " << deviceProp.major << '.' << deviceProp.minor
                  << "\n Architecture              : " << getArch(deviceProp)
                  << "\n Device clock rate         : " << deviceProp.clockRate/1000 << " MHz"
                  << "\n Memory clock rate         : " << deviceProp.memoryClockRate/1000 << " MHz"
                  << "\n Memory bus width          : " << deviceProp.memoryBusWidth << " bits"
                  << "\n Peak memory bandwidth     : " << (((deviceProp.memoryClockRate*2.0)*(deviceProp.memoryBusWidth/8))/1e6) << " GB/s"
                  << "\n Multiprocessor count      : " << deviceProp.multiProcessorCount
                  << "\n CUDA cores                : " << getCores(deviceProp)
                  << "\n Max global memory         : " << deviceProp.totalGlobalMem/(1024*1024) << " MB"
                  << "\n Max constant memory       : " << deviceProp.totalConstMem
                  << "\n Max shared memory/block   : " << deviceProp.sharedMemPerBlock
                  << "\n Max block dimensions      : " << deviceProp.maxThreadsDim[0] 
                                              << " x " << deviceProp.maxThreadsDim[1]
                                              << " x " << deviceProp.maxThreadsDim[2]
                  << "\n Max threads/block         : " << deviceProp.maxThreadsPerBlock
                  << "\n Max threads/multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor
                  << "\n Warp size                 : " << deviceProp.warpSize
                  << "\n Registers/block           : " << deviceProp.regsPerBlock
                  << "\n Registers/multiprocessor  : " << deviceProp.regsPerMultiprocessor
                  << "\n Max grid size             : " << deviceProp.maxGridSize[0]
                                                << "," << deviceProp.maxGridSize[1]
                                                << "," << deviceProp.maxGridSize[2]
                  << std::endl;
    }
    return 0;
}
