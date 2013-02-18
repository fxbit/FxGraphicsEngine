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

#include "includes.h"

extern "C"  {	
	// Device code
	__global__ void Triangulation(ThreadInfo* threadInfoArray, const RegionInfo *regionInfoArray, const ThreadParam param, const int RegionsNum)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i < RegionsNum){
            
		}
	}
}
