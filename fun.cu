#include <iterator>
#include <iostream>
#include <functional>

#include <cstdio>
#include <cuda.h>

#define cudaCheck(stmt) do { \
   cudaError_t err = stmt; \
   if( err != cudaSuccess ) { \
      std::clog << "Failed to run stmt [" << #stmt << "] (err=" << err << "): " << cudaGetErrorString(err) << '\n'; \
      cudaDeviceReset(); \
      return -1; \
   } \
} while(0)

struct Add
{
   __device__ float operator()( float a, float b ) const { return a+b; }
};

struct Sub
{
   __device__ float operator()( float a, float b ) const { return a-b; }
};

template < typename Operation >
__global__ void do_op( float* input, float* input2, float* output, int len
  , Operation op )
{
   if( threadIdx.x < len )
   {
      printf( "Thread %d: [%f + %f]\n", threadIdx.x, input[threadIdx.x], input2[threadIdx.x] );
      output[threadIdx.x] = op( input[threadIdx.x], input2[threadIdx.x] );
   }
}

int main( int argc, char** argv )
{
   float i1[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
   float i2[] = { .1, .2, .3, .4, .5, .6 };
   float i3[6];

   ::memset( i3, 0, sizeof(i3) );

   float *d1,*d2,*d3;

   cudaDeviceReset();

   cudaCheck(cudaMalloc( (void**)&d1, 6 * sizeof(float) ));
   cudaCheck(cudaMalloc( (void**)&d2, 6 * sizeof(float) ));
   cudaCheck(cudaMalloc( (void**)&d3, 6 * sizeof(float) ));

   cudaCheck(cudaMemcpy( d1, i1, 6*sizeof(float), cudaMemcpyHostToDevice ));
   cudaCheck(cudaMemcpy( d2, i2, 6*sizeof(float), cudaMemcpyHostToDevice ));

   for( int i = 0; i < 6; ++i )
      std::cout << "i1[" << i << "] = " << i1[i] << std::endl;

   dim3 grid( 1, 1, 1 );
   dim3 block( 6, 1, 1 );
   do_op<<< grid, block >>>( d1, d2, d3, 6, Sub() );
   cudaCheck( cudaPeekAtLastError() );

   std::cout << "Before:\n";
   std::copy( i3, i3+6, std::ostream_iterator<float>(std::cout,"\n") );

   cudaCheck(cudaMemcpy( i3, d3, 6*sizeof(float), cudaMemcpyDeviceToHost ));

   std::cout << "After:\n";
   std::copy( i3, i3+6, std::ostream_iterator<float>(std::cout,"\n") );

   cudaCheck(cudaFree( d1 ));
   cudaCheck(cudaFree( d2 ));
   cudaCheck(cudaFree( d3 ));

   cudaDeviceReset();

   return 0;
}

