
#include "includes.h"

#ifndef H_BOUNDARY_UTILS
#define H_BOUNDARY_UTILS

// ------------ InitNewBoundaryNode ------------ //

// init a new HalfEdge with know startVertex and return the id
__device__
uint InitNewBoundaryNode(const DATA_TYPE *VertexList, 
                         const HalfEdge *HEList,
                         ThreadInfo *threadInfo,
                         uint halfEdgeID, 
                         BoundaryNode *newBN)
{
    // get and inc the last id for halfEdge
    uint newBN_ID = threadInfo->lastBoundaryNodeID;
    threadInfo->lastBoundaryNodeID ++;

    // fill the fields
    newBN->halfEdgeID = halfEdgeID;

    // link the node with infinity...
    newBN->PrevNodeID = UNSET;
    newBN->NextNodeID = UNSET;
    
    // return the new newBN_ID
    return newBN_ID;
}














#endif /* H_BOUNDARY_UTILS */





