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


#define uint unsigned int
#define DATA_TYPE float2

#include "includes.h"
#include "TriangleUtils.h"
#include "BoundaryUtils.h"

__device__
void Init(DATA_TYPE* VertexList, 
          HalfEdge* HEList,
          ThreadInfo *threadInfo)
{
	uint bn1_ID, bn2_ID, bn3_ID;
	BoundaryNode bn1,bn2,bn3;
	HalfEdge he1,he2,he3;
	uint he1ID, he2ID, he3ID;
	uint offset =  threadInfo->offsetVertexList;
	bool ccw;
	
	// get the 3 first points and create a triangle
	// create the first triangle
	CreateTriangle(VertexList, HEList, threadInfo
                   offset, offset+1 , offset+2 ,// the 3 first vertex
				   &he1, he1ID,			// get the first he
				   &he2, he2ID,			// get the sec he
				   &he3, he3ID, 		// get the third he
			       true, ccw);	        // store the results
    
	// create a new face that start from HalfEdge1ID
	CreateFace(he1ID,  threadInfo);
	
	// create and link the boundary list   
	{
		
		// read again the face that created from the 0,1,2 vertex ... CCW problems
		bn1_ID = InitNewBoundaryNode(he1ID, bn1, threadInfo);			//  init the root bn
		bn2_ID = InitNewBoundaryNode(he1.nextEdgeID, bn2, threadInfo);	//  init the second bn
		he2 = GetHalfEdge(he1.nextEdgeID, threadInfo);					//  move to the next he
		bn3_ID = InitNewBoundaryNode(he2.nextEdgeID, bn3, threadInfo);	//  init the third bn
		he3 = GetHalfEdge(he2.nextEdgeID, threadInfo);					//  move to the next he

		// set the root bn to 1
		threadInfo.boundaryNodeRootID = bn1_ID;

		// link the root with the next node
		bn1.NextNodeID = bn2_ID;
		bn2.PrevNodeID = bn1_ID;

		// link the root with the next node
		bn2.NextNodeID = bn3_ID;
		bn3.PrevNodeID = bn2_ID;

		// link the first with the last one
		bn3.NextNodeID = bn1_ID;
		bn1.PrevNodeID = bn3_ID;


		// store the new bn
		SetBoundaryNode(bn1 , bn1_ID, threadInfo);
		SetBoundaryNode(bn2 , bn2_ID, threadInfo);
		SetBoundaryNode(bn3 , bn3_ID, threadInfo);

	}

}


// ---------------------------------------------------------------------------------

  extern "C"
__global__ void Triangulation(ThreadInfo* threadInfoArray, 
                              const RegionInfo *regionInfoArray, 
                              const ThreadParam param, 
                              const int RegionsNum)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < RegionsNum){
        ThreadInfo threadInfo;
        RegionInfo regionInfo           = regionInfoArray[i];
        
        threadInfo.threadID             =  i;
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

    }
    
    // init the triangulation by create the first triangle
    Init(&threadInfo);
    
    // save the results back to array
    threadInfoArray[i] = threadInfo;

}
