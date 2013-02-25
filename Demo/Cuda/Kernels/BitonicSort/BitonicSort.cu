//Includes for IntelliSense 
#define _SIZE_T_DEFINED
#ifndef __CUDACC__
#define __CUDACC__
#endif
#ifndef __cplusplus
#define __cplusplus
#endif

#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>

typedef unsigned int uint;

//--------------------------------------------------------------------------------------
// Struct Defines
//--------------------------------------------------------------------------------------

typedef struct{
    uint g_iLevel;
    uint g_iLevelMask;
    uint g_iWidth;
    uint g_iHeight;
    uint g_iField;
} settings_t;

#define DATA_TYPE float2

#define BITONIC_BLOCK_SIZE 1024
//#define BITONIC_BLOCK_SIZE 128

#define TRANSPOSE_BLOCK_SIZE 8
//#define TRANSPOSE_BLOCK_SIZE 8


//--------------------------------------------------------------------------------------
// Bitonic Sort Compute Shader
//--------------------------------------------------------------------------------------

__device__ bool compare_l(DATA_TYPE A, DATA_TYPE B,const uint element) { return (element == 0)? (A.x<B.x) : (A.y<B.y);}
__device__ bool compare_g(DATA_TYPE A, DATA_TYPE B,const uint element) { return (element == 0)? (A.x>B.x) : (A.y>B.y);}
__device__ bool compare_le(DATA_TYPE A, DATA_TYPE B,const uint element) { return (element == 0)? (A.x<=B.x) : (A.y<=B.y);}
__device__ bool compare_ge(DATA_TYPE A, DATA_TYPE B,const uint element) { return (element == 0)? (A.x>=B.x) : (A.y>=B.y);}


extern "C" __global__ 
void BitonicSort( const settings_t settings,
                  DATA_TYPE *Data )
{		
    __shared__ DATA_TYPE shared_data[BITONIC_BLOCK_SIZE];
    
    const uint X = blockIdx.x * blockDim.x + threadIdx.x;
    const uint GI = threadIdx.x;
    
    // Load shared data
    shared_data[GI] = Data[X];
    
    __syncthreads();
    
    // Sort the shared data
    for (unsigned int j = settings.g_iLevel >> 1 ; j > 0 ; j >>= 1)
    {
        bool b1 = compare_le(shared_data[GI & (~j)], shared_data[GI | j], settings.g_iField);
        bool b2 = ((settings.g_iLevelMask & X) != 0);
        DATA_TYPE result = ( b1 == b2 )? shared_data[GI ^ j] : shared_data[GI];
        __syncthreads();
        shared_data[GI] = result;
        __syncthreads();
    }
    
    // Store shared data
    Data[X] = shared_data[GI];
}



//--------------------------------------------------------------------------------------
// Matrix Transpose Compute Shader
//--------------------------------------------------------------------------------------

extern "C" __global__ 
void MatrixTranspose( const settings_t settings,
                      const DATA_TYPE *Input,
                      DATA_TYPE *Output)
{
    __shared__ DATA_TYPE transpose_shared_data[TRANSPOSE_BLOCK_SIZE * TRANSPOSE_BLOCK_SIZE];
    
    const uint Input_X = blockIdx.x * blockDim.x + threadIdx.x;
    const uint Input_Y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint GI = threadIdx.y * TRANSPOSE_BLOCK_SIZE + threadIdx.x;
    transpose_shared_data[GI] = Input[Input_Y * settings.g_iWidth + Input_X];
    
    __syncthreads();
    
    uint X = Input_Y - threadIdx.y + threadIdx.x;
    uint Y = Input_X - threadIdx.x + threadIdx.y;
    uint OIndex = Y * settings.g_iHeight + X;
    uint SIndex =  threadIdx.x * TRANSPOSE_BLOCK_SIZE + threadIdx.y;
    Output[OIndex] = transpose_shared_data[SIndex];
}
