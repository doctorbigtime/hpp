#include <iterator>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <stdint.h>

#define BLOCK_SIZE 32
#define SECTION_SIZE (BLOCK_SIZE*2)

#define cudaCheck(stmt) do {                               \
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
      #ifdef DEBUG
      printf( "(%d:%d) output[%d] = tmp[%d] (%f)\n", blockIdx.x, t, i, t, tmp[t] );
      #endif
      output[i] = tmp[t];
      if( i+blockDim.x < len )
      {
         #ifdef DEBUG
         printf( "(%d:%d) (2) output[%d] = tmp[%d] (%f) (blockDim=%d)\n", blockIdx.x, t
            , i+blockDim.x, t+blockDim.x, tmp[t+blockDim.x], blockDim.x );
         #endif
         output[i+blockDim.x] = tmp[t+blockDim.x];
      }
   }
   if( t == blockDim.x - 1 && sums )
   {
      //printf( "(%d:%d) assigning sum (%f) to sums\n", blockIdx.x, t, tmp[SECTION_SIZE-1] );
      sums[blockIdx.x] = tmp[SECTION_SIZE-1];
   }
}

/**
* @input and @output must already be in device memory.
*/
template < typename T, typename Operator >
void scanner( T* input, T* output, int numElements, Operator op ) 
{
   int numBlocks( ::ceil(static_cast<double>(numElements)/SECTION_SIZE) );

   std::cout << "scanner: numBlocks: " << numBlocks << "\n";

   int iterations = ::ceil(static_cast<double>(numBlocks)/0x400);
   int remainder  = numBlocks%0x400;
   for( int itr = 0; itr < iterations; ++itr )
   {
      T* sums;
      cudaCheck(cudaMalloc( (void**)&sums, numBlocks*sizeof(T) ));

      dim3 dimGrid( itr==iterations-1? remainder : numBlocks );
      scan<<< dimGrid, BLOCK_SIZE >>>( input, output, numElements, sums, op );
      cudaCheck( cudaGetLastError() );
      cudaDeviceSynchronize();

      #ifdef DEBUG
      {
         std::vector<T> s( numElements );
         cudaCheck(cudaMemcpy(&s[0],output,numElements*sizeof(T), cudaMemcpyDeviceToHost ));
         std::cout << " ============= Temporary output =============\n";
         std::copy( s.begin(), s.end()
            , std::ostream_iterator<T>(std::cout," ") );
         std::cout << "\n ============================================\n";
      }
      #endif

      if( numElements > SECTION_SIZE )
      {
      #ifdef DEBUG
         {
            std::vector<T> s( numBlocks );
            cudaCheck(cudaMemcpy( &s[0], sums, numBlocks*sizeof(T), cudaMemcpyDeviceToHost ));
            std::cout << " =========== Sums pre scan ============= \n";
            std::copy( s.begin(), s.end()
            , std::ostream_iterator<T>(std::cout," ") );
            std::cout << "\n ============================================\n";
         }
      #endif
         T* scannedSums;
         cudaCheck( cudaMalloc( (void**)&scannedSums, numBlocks*sizeof(T) ) );
         cudaMemset( scannedSums, 0, numBlocks*sizeof(T) );
         scanner( sums, scannedSums, numBlocks, op );
      #ifdef DEBUG
            std::vector<T> s( numBlocks );
            cudaCheck(cudaMemcpy( &s[0], scannedSums, numBlocks*sizeof(T), cudaMemcpyDeviceToHost ));
            std::cout << " =========== Sums post scan ============= \n";
            std::copy( s.begin(), s.end()
            , std::ostream_iterator<T>(std::cout," ") );
            std::cout << "\n ============================================\n";
      #endif
         int remaining = numElements - SECTION_SIZE;
         for( int i = 1; i < numBlocks; ++i )
         {
            dim3 dGrid( ceil((double)SECTION_SIZE/BLOCK_SIZE) );
            dim3 dBlock( BLOCK_SIZE );
            apply<<< dGrid, dBlock >>>
               ( output+(SECTION_SIZE*i)
               , remaining < SECTION_SIZE ? remaining : SECTION_SIZE
               , op
               , scannedSums+i-1 );
            cudaCheck( cudaGetLastError() );
            cudaDeviceSynchronize();
            remaining -= SECTION_SIZE;
         }
         cudaCheck( cudaFree(scannedSums) );
      }
      cudaCheck( cudaFree(sums) );
   }
}

struct Adder
{
   template < typename T >
   __device__ T operator()( T a, T b ) const { return a+b; }
};

template < typename T >
std::vector<T> importVector( const std::string& filename )
{
   std::ifstream ifs( filename.c_str() );
   if( !ifs ) return std::vector<T>();
   std::istream_iterator<T> it( ifs );
   std::vector< T > v;
   for( ; std::istream_iterator<T>() != it; ++it )
      v.push_back( *it );
      return v;
}

int main(int argc, char ** argv) 
{
   uint64_t* deviceInput;
   uint64_t* deviceOutput;
   int numElements; // number of elements in the list

   std::vector<uint64_t> hostOutput;
   std::vector<uint64_t> hostInput(importVector<uint64_t>(argv[1]));
   numElements = (int)hostInput.size();

   std::cout << "numElements: " << numElements << std::endl;

   hostOutput.resize( numElements );

   cudaCheck(cudaMalloc((void**)&deviceInput
   , numElements*sizeof(uint64_t)));
   cudaCheck(cudaMalloc((void**)&deviceOutput
   , numElements*sizeof(uint64_t)));
   cudaCheck(cudaMemset(deviceOutput, 0
   , numElements*sizeof(uint64_t)));

   cudaCheck(cudaMemcpy(deviceInput, &(hostInput[0])
   , numElements*sizeof(uint64_t), cudaMemcpyHostToDevice));

   std::cout << "Launching scanner.\n";
   scanner( deviceInput, deviceOutput, numElements, Adder() );

   cudaCheck(cudaMemcpy(&(hostOutput[0]), deviceOutput
   , numElements*sizeof(uint64_t), cudaMemcpyDeviceToHost));

   if( numElements < 100 )
      std::copy( hostOutput.begin(), hostOutput.end()
      , std::ostream_iterator<uint64_t>(std::cout,"\n") );
   else std::cout << "Last element: " << hostOutput.back() << "\n";

   cudaFree(deviceInput);
   cudaFree(deviceOutput);

   return 0;
}

