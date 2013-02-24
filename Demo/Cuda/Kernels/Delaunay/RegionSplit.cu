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

/* --------------------------------------------------------------------------------- */

extern "C"
__global__ void splitRegionH(const DATA_TYPE *vertex,
                           RegionInfo *regionInfo,
                           uint HorizontalRegionNum,
                           uint SplitOffset,
                           uint NumElements)
{
    const uint i = blockDim.x * blockIdx.x + threadIdx.x;
    RegionInfo r;
    
    // limit check
    if( i >= HorizontalRegionNum)
        return;
    
    // set the first region to start from zero
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
                           
/* --------------------------------------------------------------------------------- */

extern "C"
__global__ void splitRegionV_phase1(const DATA_TYPE *vertex,
                             const RegionInfo *regionInfoH,
                             RegionInfo *regionInfoV,
                             uint HorizontalRegionNum,
                             uint VerticalRegionNum,
                             uint SplitOffset)
{
    const uint v = blockDim.x * blockIdx.x + threadIdx.x;
    const uint h = blockDim.y * blockIdx.y + threadIdx.y;
    
    RegionInfo regionH = regionInfoH[h];
    RegionInfo r;
    
    // limit check
    if( h >= HorizontalRegionNum || v >= VerticalRegionNum)
        return;
    
    // set the first region to start from the start of the 
    // horizontal zone
    if(v==0){
        r.VertexOffset = regionH.VertexOffset;
        regionInfoV[h*VerticalRegionNum + v] = r;
        return;
    }
    
    // find the start point of the split
    uint index = v*SplitOffset + regionH.VertexOffset;
    uint maxNum = regionH.VertexOffset + regionH.VertexNum;
    
    DATA_TYPE data = vertex[index];
    
    // find the bigger index that we have the same y
    while( (index+1) < maxNum &&
            vertex[index+1].x == data.x)
    {
        data = vertex[++index];
    }
    
    // update the index
    r.VertexOffset = index;
    regionInfoV[h*VerticalRegionNum + v] = r;
}

extern "C"
__global__ void splitRegionV_phase2( const RegionInfo *regionInfoH,                         
                                     RegionInfo *regionInfoV,
                                     uint VerticalRegionNum,
                                     uint NumRegions)
{
    const uint i = blockDim.x * blockIdx.x + threadIdx.x;
    
    // limit check
    if( i >= NumRegions)
        return;
    
    RegionInfo regionI, regionI_1;
    
    regionI = regionInfoV[i];
    if(i+1<NumRegions){
        regionI_1 = regionInfoV[i+1];
	}else{
		int index = (int)floor((float)i/VerticalRegionNum)+1;
        regionI_1 = regionInfoH[index];
    }   
    __syncthreads();

    regionI.VertexNum = regionI_1.VertexOffset - regionI.VertexOffset;
        
    __syncthreads();
    
    regionInfoV[i] = regionI;
}




