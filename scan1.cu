#include <iterator>
#include <iostream>

#include    <wb.h>

#define BLOCK_SIZE 32 //@@ You can change this
#define SECTION_SIZE (BLOCK_SIZE*2) 

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            std::cerr << "Failed to run stmt " #stmt       \
            << " on line " << __LINE__                     \
            << ": " << cudaGetErrorString(err) << "\n";    \
            exit(-1);                                      \
        }                                                  \
    } while(0)

template < typename T, typename Operator >
__global__ void apply( T* input, int len, Operator op, const T* val )
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if( i < len )
      input[i] = op( input[i], *val );
}

template < typename T, typename Operator >
__global__ void scan(T* input, T* output, int len, T* sums, Operator op )
{
   __shared__ T tmp[ SECTION_SIZE ];

   int t = threadIdx.x;
   int i = blockIdx.x * SECTION_SIZE + t;

   tmp[ t ] = .0;
   tmp[ t+blockDim.x ] = .0;
   if( i < len )
   {
      tmp[t] = input[i];
      if( i+blockDim.x < len )
         tmp[t+blockDim.x] = input[i+blockDim.x];

      for( int stride = 1; stride <= BLOCK_SIZE; stride *= 2 )
      {
         int index = (t+1)*stride*2-1;
         if( index < SECTION_SIZE )
            tmp[index] = op( tmp[index], tmp[index-stride] );
         __syncthreads();
      }
      for( int stride = BLOCK_SIZE/2; stride > 0; stride /= 2 )
      {
         __syncthreads();
         int index = (t+1)*stride*2-1;
         if( index+stride < SECTION_SIZE )
            tmp[index+stride] = op( tmp[index+stride], tmp[index] );
      }
      __syncthreads();
      output[i] = tmp[t];
      if( i+blockDim.x < len )
         output[i+blockDim.x] = tmp[t+blockDim.x];
   }
   if( t == blockDim.x - 1 && sums )
      sums[blockIdx.x] = tmp[SECTION_SIZE-1];
}

template < typename T, typename Operator >
void scanner( T* input, T* output, int numElements, Operator op ) 
{
   int numBlocks( ::ceil(static_cast<double>(numElements)/SECTION_SIZE) );

   int iterations = ::ceil(static_cast<double>(numBlocks)/0x400);
   int remainder  = numBlocks%0x400;
   for( int itr = 0; itr < iterations; ++itr )
   {
      T* sums;
      wbCheck(cudaMalloc( (void**)&sums, numBlocks*sizeof(T) ));

      dim3 dimGrid( itr==iterations-1? remainder : numBlocks );
      scan<<< dimGrid, BLOCK_SIZE >>>( input, output, numElements, sums, op );
      wbCheck( cudaGetLastError() );
      cudaDeviceSynchronize();

      if( numElements > SECTION_SIZE )
      {
         T* sumSums;
         wbCheck( cudaMalloc( (void**)&sumSums, numBlocks*sizeof(T) ) );
         cudaMemset( sumSums, 0, numBlocks*sizeof(T) );
         scanner( sums, sumSums, numBlocks, op );
         int remaining = numElements - SECTION_SIZE;
         for( int i = 1; i < numBlocks; ++i )
         {
            dim3 dGrid( ceil((double)SECTION_SIZE/BLOCK_SIZE) );
            dim3 dBlock( BLOCK_SIZE );
            apply<<< dGrid, dBlock >>>
               ( output+(SECTION_SIZE*i)
               , remaining < SECTION_SIZE ? remaining : SECTION_SIZE
               , op
               , sumSums+i-1 );
            wbCheck( cudaGetLastError() );
            cudaDeviceSynchronize();
            remaining -= SECTION_SIZE;
         }
         wbCheck( cudaFree(sumSums) );
      }
      wbCheck( cudaFree(sums) );
   }
}

struct Adder
{
   __device__ float operator()( float a, float b ) const { return a+b; }
};

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    wbTime_start(Compute, "Performing CUDA computation");

    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
    scanner( deviceInput, deviceOutput, numElements, Adder() );

    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}

