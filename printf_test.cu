#include "stdio.h"
#include <cuda.h>

//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
//#  error "printf not defined for CUDA_ARCH" ##__CUDA_ARCH__
//#endif

__global__ void helloCuda( float f )
{
   printf( "Hello thread %d, f = %f\n", threadIdx.x, f );
}

#ifdef __CUDACC__
#warning cuda cc
#endif

int main( int argc, char** argv )
{
   helloCuda<<< 1, 5 >>>( 3.14159 );
   cudaDeviceSynchronize();
   return 0;
}
   
