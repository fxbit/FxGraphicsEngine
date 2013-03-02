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
#include "IOBuffer.h"
#include "HalfEdgeUtils.h"
#include "FaceUtils.h"
#include "TriangleUtils.h"
#include "BoundaryUtils.h"

__device__
void Init(const DATA_TYPE   *VertexList, 
          HalfEdge          *HEList,
          BoundaryNode      *BoundaryList,
          Face              *FaceList,
          ThreadInfo        *threadInfo)
{
    uint bn1_ID, bn2_ID, bn3_ID;
    BoundaryNode bn1,bn2,bn3;
    HalfEdge he1,he2,he3;
    uint3 heID;
    uint offset =  threadInfo->offsetVertexList;
    bool ccw;
    
    // get the 3 first points and create a triangle
    // create the first triangle
    CreateTriangle<true>(VertexList, HEList, threadInfo, 
                         offset, offset+1 , offset+2 ,// the 3 first vertex
                         &he1, &he2, &he3,
                         &heID,
                         &ccw);
    
    // create a new face that start from HalfEdge1ID
    CreateFace(HEList,
               FaceList,
               heID.x,
               threadInfo);
    
    // create and link the boundary list   
    {
        
        // read again the face that created from the 0,1,2 vertex ... CCW problems
        bn1_ID = InitNewBoundaryNode(VertexList, HEList, threadInfo,
                                     heID.x, &bn1);			            //  init the root bn
        bn2_ID = InitNewBoundaryNode(VertexList, HEList, threadInfo,
                                     he1.nextEdgeID, &bn2);	            //  init the second bn
        he2    = HEList[he1.nextEdgeID];                 				//  move to the next he
        bn3_ID = InitNewBoundaryNode(VertexList, HEList, threadInfo,
                                     he2.nextEdgeID, &bn3);	            //  init the third bn
        he3    = HEList[he2.nextEdgeID];					            //  move to the next he

        // set the root bn to 1
        threadInfo->boundaryNodeRootID = bn1_ID;

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
        SetBoundaryNode(BoundaryList, &bn1, bn1_ID,  threadInfo);
        SetBoundaryNode(BoundaryList, &bn2, bn2_ID,  threadInfo);
        SetBoundaryNode(BoundaryList, &bn3, bn3_ID,  threadInfo);

    }

}


// ---------------------------------------------------------------------------------

extern "C"
__global__ void Triangulation(const DATA_TYPE   *VertexList,
                              HalfEdge          *HEList,
                              BoundaryNode      *BoundaryList,
                              Face              *FaceList,
                              ThreadInfo        *threadInfoArray,
                              const RegionInfo  *regionInfoArray,
                              const ThreadParam  param,
                              const int          RegionsNum)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= RegionsNum)
        return;
    
    ThreadInfo threadInfo;
    RegionInfo regionInfo           = regionInfoArray[i];
    
    threadInfo.threadID             =  i;
    threadInfo.offsetFaceList		=  i * param.maxFacesPerThread;
    threadInfo.offsetHalfEdgeList	=  i * param.maxHalfEdgePerThread;
    threadInfo.offsetVertexList		=  regionInfo.VertexOffset;
    threadInfo.offsetBoundaryList	=  i * param.maxBoundaryNodesPerThread;
    threadInfo.offsetDNStack		=  i * MAX_FACE_CORRECTIONS;

    threadInfo.lastFaceID			=  make_uint2(0,threadInfo.offsetFaceList); 	// no face yet
    threadInfo.lastHalfEdgeID		=  threadInfo.offsetHalfEdgeList; 	// no he yet
    threadInfo.lastBoundaryNodeID	=  0; 	// no bn yet
    threadInfo.boundaryNodeRootID	=  0; 	// set the node root to unset;
    threadInfo.endDNOfStack 		=  0; 	// no DN in stack yet
    threadInfo.startDNOfStack 		=  0; 	// no DN in stack yet
    threadInfo.numDNinStack			=  0; 	// no DN in stack yet

    // init the triangulation by create the first triangle
    Init(VertexList, 
         HEList,
         BoundaryList,
         FaceList,
         &threadInfo);
    
    // save the results back to array
    threadInfoArray[i] = threadInfo;

}
