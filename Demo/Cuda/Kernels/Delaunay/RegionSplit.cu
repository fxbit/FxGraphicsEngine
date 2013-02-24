//Includes for IntelliSense 
#define _SIZE_T_DEFINED
#ifndef __CUDACC__
#define __CUDACC__
#endif
#ifndef __cplusplus
#define __cplusplus
#endif


#define uint unsigned int
#define DATA_TYPE float2

#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>

#include "includes.h"

extern "C"
__global__ void splitRegionH(const DATA_TYPE *vertex,
                           RegionInfo *regionInfo,
                           uint HorizontalRegionNum,
                           uint SplitOffset,
                           uint NumElements)
{
    const uint i = blockDim.x * blockIdx.x + threadIdx.x;
    
    RegionInfo r;
    
    if(i==0){
        r.VertexOffset = 0;
        regionInfo[i] = r;
        return;
    }
    
    // find the start point of the split
    uint index = i*SplitOffset;
    DATA_TYPE data = vertex[index];
    
    // find the bigger index that we have the same y
    while( (index+1) < NumElements &&
           vertex[index+1].y == data.y)
    {
        data = vertex[++index];
    }
    
    // update the index
	r.VertexOffset = index;
	regionInfo[i] = r;
}
                           
