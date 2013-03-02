#include "includes.h"

#ifndef CUH_HALF_EDGE_UTILS
#define CUH_HALF_EDGE_UTILS



// ------------ InitNewHalfEdge  ------------ //

// init a new HalfEdge with know startVertex and return the id
__device__
uint InitNewHalfEdge(uint        StartVertexID, 
                     HalfEdge   *newHE, 
                     ThreadInfo *threadInfo)
{
    // get and inc the last id for halfEdge
    uint newHeID = threadInfo->lastHalfEdgeID;
    threadInfo->lastHalfEdgeID++;

    // fill the fielts
    newHE->startVertexID = StartVertexID;

    // link the edge with infinity...
    newHE->twinEdgeID = UNSET;
    newHE->nextEdgeID = UNSET;
    
    // set the face it to unset
    newHE->faceID = make_uint2(UNSET,UNSET);
    
    // return the new halfEdgeID
    return newHeID;
}




#endif /* CUH_HALF_EDGE_UTILS */
