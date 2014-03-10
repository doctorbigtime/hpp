#include    <wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float * A, float * B, float * C,
                      int numARows, int numAColumns, int numBColumns )
 {
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
   __shared__ float sharedA[ TILE_WIDTH ][ TILE_WIDTH ];
   __shared__ float sharedB[ TILE_WIDTH ][ TILE_WIDTH ];

   int bx = blockIdx.x;  int by = blockIdx.y;
   int tx = threadIdx.x; int ty = threadIdx.y;

   int Row = by * blockDim.y + ty;
   int Col = bx * blockDim.x + tx;
   float Cvalue = .0;

   for( int t = 0; t < (numAColumns-1)/TILE_WIDTH+1; ++t )
   {
      if( Row < numARows && t*TILE_WIDTH+tx < numAColumns )
         sharedA[ ty ][ tx ] = A[ Row*numAColumns + t*TILE_WIDTH+tx ];
      else sharedA[ ty ][ tx ] = .0;
      if( t*TILE_WIDTH+ty < numAColumns && Col < numBColumns )
         sharedB[ ty ][ tx ] = B[ (t*TILE_WIDTH+ty)*numBColumns + Col ];
      else sharedB[ ty ][ tx ] = .0;
      __syncthreads();

      for( int i = 0; i < TILE_WIDTH; ++i )
         Cvalue += sharedA[ty][i] * sharedB[i][tx];
      __syncthreads();
   }
   if( Row < numARows && Col < numBColumns )
      C[Row*numBColumns+Col] = Cvalue;
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    
    size_t bytesA, bytesB, bytesC;
    bytesA = numARows*numAColumns*sizeof(float);
    bytesB = numBRows*numBColumns*sizeof(float);
    bytesC = numCRows*numCColumns*sizeof(float);

    //@@ Allocate the hostC matrix
    hostC = (float*) calloc( numCRows * numCColumns, sizeof(float) );
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    wbCheck(cudaMalloc((void**)&deviceA, bytesA));
    wbCheck(cudaMalloc((void**)&deviceB, bytesB));
    wbCheck(cudaMalloc((void**)&deviceC, bytesC));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    wbCheck(cudaMemcpy(deviceA,hostA,bytesA,cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(deviceB,hostB,bytesB,cudaMemcpyHostToDevice));

    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 dimGrid( (numBColumns-1)/TILE_WIDTH+1, (numARows-1)/TILE_WIDTH+1, 1 );
    dim3 dimBlock( TILE_WIDTH, TILE_WIDTH, 1 );

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    matrixMultiplyShared<<< dimGrid, dimBlock >>>( deviceA, deviceB, deviceC
                        , numARows, numAColumns, numBColumns );

    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    wbCheck(cudaMemcpy( hostC, deviceC, bytesC, cudaMemcpyDeviceToHost ));

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    wbCheck( cudaFree(deviceA) );
    wbCheck( cudaFree(deviceB) );
    wbCheck( cudaFree(deviceC) );

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}

