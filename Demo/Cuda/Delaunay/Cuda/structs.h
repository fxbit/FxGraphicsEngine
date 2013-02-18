
#include "includes.h"

#ifndef H_STRUCTS
#define H_STRUCTS

#define uint unsigned int

typedef struct{

// max faces per thread
	uint maxFacesPerThread;

// max Half edge per thread
	uint maxHalfEdgePerThread;

// max boundary nodes per thread
	uint maxBoundaryNodesPerThread;

// num of the regions
	uint RegionsNum;

} ThreadParam;

// ------------ HalfEdge  ------------ //

typedef struct{

	// The id of the thread
	uint threadID;

	// offset for the face list
	uint offsetFaceList;

	// offset for the Half edge list
	uint offsetHalfEdgeList;

	// offset for the Vertex list
	uint offsetVertexList;

	// offset for the Boundary list
	uint offsetBoundaryList;

	// offset for the delaunay node 
	uint offsetDNStack;
	
	// The Next Face ID (max face id)
	uint lastFaceID;

	// The Next HalfEdge ID (max HalfEdge id)
	uint lastHalfEdgeID;

	// The Next Boundary Node ID (max Boundary Node id)
	uint lastBoundaryNodeID;

	// The rood index of the boundary
	uint boundaryNodeRootID;
	
	// the start of the DelaunayNode in the stack
	uint startDNOfStack;
	
	// the end of the DelaunayNode in the stack
	uint endDNOfStack;

	// the num of node in the stack
	uint numDNinStack;
	
	// Boundary Min/Max {
	
    /// The Max boundary node in the X axis
    uint Boundary_X_MaxID;
    
    /// The Max boundary node in the Y axis
    uint Boundary_Y_MaxID;
    
    /// The Min boundary node in the X axis
    uint Boundary_X_MinID;
    
    /// The Min boundary node in the Y axis
    uint Boundary_Y_MinID;
    
	//}
	
    // emergency HE id
    uint LeftLastMergingHEID;
    uint RightLastMergingHEID;
    uint LeftFirstMergingHEID;
    uint RightFirstMergingHEID;
} ThreadInfo;

// ------------ HalfEdge  ------------ //

typedef struct {
	//	Start Vertex index
	uint startVertexID;

	//	twin edge of the face
	uint twinEdgeID;

	//  next edge in the face
	uint nextEdgeID;
	
	//  face this edge belong
	uint2 faceID;
} HalfEdge;

// ------------ BoundaryNodes  ------------ //

typedef struct {
	//	The next node on the boundary
	uint PrevNodeID;

	//	The prev node on the boundary
	uint NextNodeID;

	//  The edge in the boundary
	uint halfEdgeID;
} BoundaryNode;

// ------------ Circle  ------------ //

typedef struct {
	
	// Center
	float2 center;

	// radius 
	float radius;

} Circle;


// ------------ Face  ------------ //

typedef  struct {

	//  Start of the Face 
	uint  halfEdgeID; 
	
#ifdef USE_BOUNDARY
#ifdef USE_CIRCLE_TEST

	//  Boundary circle
	Circle boundary;

#else

	// The Min point of the boundary
	float2 Min;

	// The Max point of the boundary
	float2 Max;

#endif
#endif 
} Face;

// ------------ Region Info  ------------ //

typedef struct {

	// the offset of the vertex
	uint VertexOffset;
	
	// the number of the vertex
	uint VertexNum;

}RegionInfo;

// ------------ Region Info  ------------ //

typedef struct  {

	// the start half edge
	uint RootHalfEdgeID;
	
	// the half edge of the next face
	uint NextFaceHalfEdgeID;
	
}DelaunayNode;

typedef struct{
	
	// the id of the 2 sides of the 
	// x: left
	// y: right
	uint2 RegionSidesID; 
	
}MergeVParams;

// ------------ Stack struct  ------------ //

typedef struct{
	uint start;
	uint end;
	uint offset;
} stack;

// ------------ VMerge thread Info  ------------ //

typedef struct{
	
	// current thread id
	uint threadID;

	// stack for the he merging
	stack LeftHEStack;
	stack RightHEStack;
	
} MergeVInfo;


// ------------ HMerge thread Info  ------------ //

typedef struct{
	
	// current thread id
	uint threadID;

	// stack for the he merging
	stack UpHEStack;
	stack DownHEStack;
	
	// the first left Thread Info
	uint UpFirstThreadID;
	uint DownFirstThreadID;
	uint UpLastThreadID;
	uint DownLastThreadID;
} MergeHInfo;


#endif /* H_STRUCTS  */
