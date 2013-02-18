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
			ThreadInfo threadInfo;
			RegionInfo regionInfo           = regionInfoArray[i];
            
			threadInfo.threadID             = i;
			threadInfo.offsetFaceList		=  i * param.maxFacesPerThread;
			threadInfo.offsetHalfEdgeList	=  i * param.maxHalfEdgePerThread;
			threadInfo.offsetVertexList		=  regionInfo.VertexOffset;
			threadInfo.offsetBoundaryList	=  i * param.maxBoundaryNodesPerThread;
			threadInfo.offsetDNStack		=  i * MAX_FACE_CORRECTIONS;
	
			threadInfo.lastFaceID			=  0; 	// no face yet
			threadInfo.lastHalfEdgeID		=  threadInfo.offsetHalfEdgeList; 	// no he yet
			threadInfo.lastBoundaryNodeID	=  0; 	// no bn yet
			threadInfo.boundaryNodeRootID	=  0; 	// set the node root to unset;
			threadInfo.endDNOfStack 		=  0; 	// no DN in stack yet
			threadInfo.startDNOfStack 		=  0; 	// no DN in stack yet
			threadInfo.numDNinStack			=  0; 	// no DN in stack yet

			// save the results back to array
			threadInfoArray[i] = threadInfo;
		}
	}
}
