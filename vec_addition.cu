// MP 1
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <wb.h>

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
   int i = threadIdx.x + blockDim.x * blockIdx.x;
   printf( "i: %d\n", i );
   if( i < len ) out[i] = in1[i] + in2[i];
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * h_A;
    float * h_B;
    float * h_C;
    float * d_A;
    float * d_B;
    float * d_C;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    h_A = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    h_B = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    size_t size = inputLength*sizeof(float);
    h_C = (float *) malloc(size);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);
    std::cout << "size: " << size << std::endl;
   
   /*for( int i = 0; i < inputLength; ++i ) std::cout << "h_A[" << i << "] = " << h_A[i] << std::endl;
   for( int i = 0; i < inputLength; ++i ) std::cout << "h_B[" << i << "] = " << h_B[i] << std::endl;*/

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    cudaError_t err;
    if( cudaSuccess != (err = cudaMalloc( (void**) &d_A, size )) )
      std::cerr << cudaGetErrorString(err) << " on line " << __LINE__ << std::endl;
    if( cudaSuccess != (err = cudaMalloc( (void**) &d_B, size )) )
      std::cerr << cudaGetErrorString(err) << " on line " << __LINE__ << std::endl;
    if( cudaSuccess != (err = cudaMalloc( (void**) &d_C, size )) )
      std::cerr << cudaGetErrorString(err) << " on line " << __LINE__ << std::endl;

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    if( cudaSuccess != (err = cudaMemcpy( d_A, h_A, size, cudaMemcpyHostToDevice )) )
      std::cerr << cudaGetErrorString(err) << " on line " << __LINE__ << std::endl;
    if( cudaSuccess != (err = cudaMemcpy( d_B, h_B, size, cudaMemcpyHostToDevice )) )
      std::cerr << cudaGetErrorString(err) << " on line " << __LINE__ << std::endl;

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
    dim3 dimGrid( ceil(inputLength/256.0), 1, 1 );
    dim3 dimBlock( 256, 1, 1 );
    
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    vecAdd<<< dimGrid, dimBlock >>>(d_A,d_B,d_C,inputLength);

    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    if( cudaSuccess != (err = cudaMemcpy( h_C, d_C, size, cudaMemcpyDeviceToHost )) )
      std::cerr << cudaGetErrorString(err) << " on line " << __LINE__ << std::endl;
    for( int i = 0; i < inputLength; ++i ) std::cout << "h_C[" << i << "] = " << h_C[i] << std::endl;

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    if( cudaSuccess != (err = cudaFree(d_A)) )
      std::cerr << cudaGetErrorString(err) << " on line " << __LINE__ << std::endl;
    if( cudaSuccess != (err = cudaFree(d_B)) )
      std::cerr << cudaGetErrorString(err) << " on line " << __LINE__ << std::endl;
    if( cudaSuccess != (err = cudaFree(d_C)) )
      std::cerr << cudaGetErrorString(err) << " on line " << __LINE__ << std::endl;

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, h_C, inputLength);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
