#include "includes.hlslh"

#ifndef CUH_HALF_EDGE_UTILS
#define CUH_HALF_EDGE_UTILS



// ------------ InitNewHalfEdge  ------------ //

// init a new HalfEdge with know startVertex and return the id
uint InitNewHalfEdge(uint StartVertexID, HalfEdge *newHE, ThreadInfo *threadInfo)
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
	newHE->faceID = UNSET;
    
	// return the new halfEdgeID
	return newHeID;
}




#endif /* CUH_HALF_EDGE_UTILS */
