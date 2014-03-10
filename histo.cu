// Histogram Equalization

#include    <wb.h>

#define cudaCheck(stmt) do {                               \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            std::cerr << "Failed to run stmt " #stmt       \
            << " on line " << __LINE__                     \
            << ": " << cudaGetErrorString(err) << "\n";    \
            exit(-1);                                      \
        }                                                  \
    } while(0)

#define BLOCK_WIDTH 8
#define HISTOGRAM_LENGTH 256

//@@ insert code herea

__global__ void toChar( const float* data, uint8_t* output, int len )
{
   int index = threadIdx.x + blockIdx.x * blockDim.x;
   output[ index ] = static_cast<unsigned char>(255*data[index]);
}

__global__ void RGB2Grayscale( const uint8_t* input, int width, int height
   , uint8_t* output )
{
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;

   if( row < height && col < width )
   {
      int idx = row*width + col;
      float r = input[ 3*idx ];
      float g = input[ 3*idx + 1 ];
      float b = input[ 3*idx + 2 ];
      output[ idx ] = static_cast<uint8_t>(0.21*r + 0.71*g + 0.07*b);
   }
}

__global__ void histogram( const uint8_t* input, int size, int* hist )
{
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   printf( "(%d:%d)\n", blockIdx.x, threadIdx.x );
   if( index < size )
   {
       printf( "(%d:%d) index (%d) - value = %d count(%d)\n", blockIdx.x, threadIdx.x, index, (int)input[index], hist[input[index]] );
       atomicAdd( &hist[input[index]], 1 );
       printf( "(%d:%d) count is now: %d\n", blockIdx.x, threadIdx.x, hist[input[index]] );
   }
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    const char * inputImageFile;

    float    *devInputImageData;
    uint8_t  *devCharImage, *devGrayscaleImageData;
    int      *devHistogram;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    hostInputImageData = wbImage_getData( inputImage );
    hostOutputImageData = wbImage_getData( inputImage );

    cudaMalloc( &devInputImageData
        , imageWidth*imageHeight*imageChannels * sizeof(float)
        );
    cudaMalloc( &devCharImage
        , imageWidth*imageHeight*imageChannels * sizeof(uint8_t)
        );
    cudaMemcpy( devInputImageData, hostInputImageData
        , imageWidth*imageHeight*imageChannels * sizeof(float)
        , cudaMemcpyHostToDevice );

    std::cout << "imageHeight: " << imageHeight
        << " imageWidth: " << imageWidth
        << std::endl;
    {
        dim3 dBlock( BLOCK_WIDTH, 1, 1 );
        dim3 dGrid( ((imageHeight*imageWidth*imageChannels)-1)/BLOCK_WIDTH+1, 1, 1 );
        toChar<<< dGrid, dBlock >>>( devInputImageData, devCharImage, imageHeight*imageWidth*imageChannels );
        cudaCheck( cudaGetLastError() );
        cudaDeviceSynchronize();
    }
    cudaMalloc( &devGrayscaleImageData
        , imageWidth*imageHeight * sizeof(uint8_t)
        );
    {
        dim3 dBlock( BLOCK_WIDTH, BLOCK_WIDTH, imageChannels );
        std::cout << "(imageWidth-1)/BLOCK_WIDTH+1: " << (imageWidth-1)/BLOCK_WIDTH+1 << std::endl;
        dim3 dGrid( (imageWidth-1)/BLOCK_WIDTH+1,(imageHeight-1)/BLOCK_WIDTH+1 );
        std::cout << "RGB dGrid.x: " << dGrid.x << " dGrid.y: " << dGrid.y << " dGrid.z: " << dGrid.z << std::endl;
        std::cout << "RGB dBlock.x: " << dBlock.x << " dBlock.y: " << dBlock.y << " dBlock.z: " << dBlock.z << std::endl;
        RGB2Grayscale<<< dGrid, dBlock >>>( devCharImage, imageWidth, imageHeight, devGrayscaleImageData );
        cudaCheck( cudaGetLastError() );
        cudaDeviceSynchronize();
    }
    // Copy to host 
    /*{
        uint8_t* grayscale = new uint8_t[ imageHeight*imageWidth ];
        cudaMemcpy( grayscale, devGrayscaleImageData, 
    }*/
    cudaMalloc( &devHistogram
        , HISTOGRAM_LENGTH * sizeof(int)
        );
    cudaCheck( cudaMemset( devHistogram, 0, HISTOGRAM_LENGTH*sizeof(int) ) );
    {
      dim3 dGrid( (imageHeight*imageWidth-1)/BLOCK_WIDTH+1 );
      dim3 dBlock( BLOCK_WIDTH );
      std::cout << "dGrid.x: " << dGrid.x << std::endl;
      histogram<<< 256, dBlock >>>( devGrayscaleImageData, imageHeight*imageWidth, devHistogram );
      cudaCheck( cudaGetLastError() );
      cudaDeviceSynchronize();
    }
    {
        int* hostHistogram = new int[ HISTOGRAM_LENGTH ];
        cudaMemcpy( hostHistogram, devHistogram, HISTOGRAM_LENGTH, cudaMemcpyDeviceToHost );
        for( int i = 0; i < HISTOGRAM_LENGTH; ++i )
        {
            std::cout << "Bucket [" << i << "] = [" << hostHistogram[i] << "]\n";
        }
    }
    /*cudaMalloc( &devOutputImageData
        , imageWidth*imageHeight*imageChannels * sizeof(float)
        );*/

    wbSolution(args, outputImage);

    //@@ insert code here

    return 0;
}

