#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

//@@ The purpose of this code is to become familiar with the submission 
//@@ process. Do not worry if you do not understand all the details of 
//@@ the code.

int main(int argc, char ** argv) {
    int deviceCount;

    //wbArg_read(argc, argv);

    cudaGetDeviceCount(&deviceCount);

    //wbTime_start(GPU, "Getting GPU Data."); //@@ start a timer

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;

        cudaGetDeviceProperties(&deviceProp, dev);

        if (dev == 0) {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
                //wbLog(TRACE, "No CUDA GPU has been detected");
                return -1;
            } else if (deviceCount == 1) {
                //@@ WbLog is a provided logging API (similar to Log4J).
                //@@ The logging function wbLog takes a level which is either
                //@@ OFF, FATAL, ERROR, WARN, INFO, DEBUG, or TRACE and a
                //@@ message to be printed.
                //wbLog(TRACE, "There is 1 device supporting CUDA");
            } else {
                //wbLog(TRACE, "There are ", deviceCount, " devices supporting CUDA");
            }
        }

        std::cout << "Device " << dev << " name: " << deviceProp.name 
                  << "\n Computational Capabilities: " << deviceProp.major << '.' << deviceProp.minor
                  << "\n Maximum global memory size: " << deviceProp.totalGlobalMem/(1024*1024) << " MB"
                  << "\n Maximum constant memory size: " << deviceProp.totalConstMem
                  << "\n Maximum shared memory size per block: " << deviceProp.sharedMemPerBlock
                  << "\n Maximum block dimensions: " << deviceProp.maxThreadsDim[0] 
                     << " x " << deviceProp.maxThreadsDim[1]
                     << " x " << deviceProp.maxThreadsDim[2]
                  << std::endl;
    }
    return 0;
}
