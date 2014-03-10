#include    <wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)


#define MASK_WIDTH  5
#define MASK_RADIUS MASK_WIDTH/2

#define O_TILE_WIDTH 32
#define BLOCK_WIDTH  O_TILE_WIDTH+MASK_WIDTH-1

//@@ INSERT CODE HERE
__global__ void imageConvolution( const float* input
   , float* output
   , const float* __restrict__ mask
   //, const float * mask
   , int height
   , int width
   , const int channels )
{
   __shared__ float i_tile[ BLOCK_WIDTH ][ BLOCK_WIDTH ][ 3 ];

   int tx = threadIdx.x;
   int ty = threadIdx.y;
   //int tz = threadIdx.z;

   int o_col = blockIdx.x * O_TILE_WIDTH + tx;
   int o_row = blockIdx.y * O_TILE_WIDTH + ty;
   //int o_cha = blockIdx.z * O_TILE_WIDTH + tz;

   int i_col = o_col - MASK_RADIUS;
   int i_row = o_row - MASK_RADIUS;
   //int i_cha = o_cha - MASK_RADIUS;

   if( (i_row >= 0 && i_row < height) && (i_col >= 0 && i_col < width) )
   {
      for( int channel = 0; channel < channels; ++channel )
      {
         i_tile[ ty ][ tx ][ channel ] = input[ (i_row*width + i_col)*channels + channel ];
         printf( "i_tile[%d][%d][%d] = [%f]\n", ty,tx,channel,i_tile[ty][tx][channel] );
      }
   }
   else
   {
      for( int channel = 0; channel < channels; ++channel )
         i_tile[ ty ][ tx ][ channel ] = .0;
   }
   __syncthreads();

   if( ty < O_TILE_WIDTH && tx < O_TILE_WIDTH )
   {
      if( o_row < height && o_col < width )
      {
         for( int channel = 0; channel < channels; ++channel )
         {
            float val = .0;
            for( int i = 0; i < MASK_WIDTH; ++i )
               for( int j = 0; j < MASK_WIDTH; ++j )
                  val += (mask[i*MASK_WIDTH+j] * i_tile[i+ty][j+tx][channel]);
            output[ (o_row*width + o_col)*channels + channel ] = val;
         }
      }
   }
   __syncthreads();
}


int main(int argc, char* argv[])
{
    wbArg_t arg;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    arg = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(arg, 0);
    inputMaskFile = wbArg_getInputFile(arg, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    //wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");

   for( int i = 0; i < 5; ++i )
   {
      for( int j = 0; j < 5; ++j )
         std::cout << "mask[" << i << "][" << j << "] = [" 
            << hostMaskData[i*MASK_WIDTH+j] << "]\n";
   }

    /*
    for( int y = 0; y < imageHeight; ++y )
    {
      for( int x = 0; x < imageWidth; ++x )
         for( int c = 0; c < imageChannels; ++c )
            std::cout << y << ":" << x << ":" << (0 == c ? 'R' : 1 == c ? 'G' : 'B')
            << " = [" << hostInputImageData
               [ y*imageWidth*imageChannels
               + x*imageChannels
               + c ]
            << "] ";
      std::cout << '\n';
    }
    */

    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");

    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
    dim3 dimBlock( BLOCK_WIDTH, BLOCK_WIDTH, 1 );
    dim3 dimGrid( (imageWidth-1)/O_TILE_WIDTH+1, (imageHeight-1)/O_TILE_WIDTH+1, 1 );

    imageConvolution<<< dimGrid, dimBlock >>>( deviceInputImageData
      , deviceOutputImageData
      , deviceMaskData
      , imageHeight, imageWidth, imageChannels );

    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    /*
    for( int y = 0; y < imageHeight; ++y )
    {
      for( int x = 0; x < imageWidth; ++x )
         for( int c = 0; c < imageChannels; ++c )
            std::cout << y << ":" << x << ":" << (0 == c ? 'R' : 1 == c ? 'G' : 'B')
            << " = [" << hostOutputImageData
               [ y*imageWidth*imageChannels
               + x*imageChannels
               + c ]
            << "] ";
      std::cout << '\n';
    }
    */

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(arg, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}

