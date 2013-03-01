#include "includes.hlslh"

#ifndef H_TRIANGLE_UTILS
#define H_TRIANGLE_UTILS

bool InCircle ( float2 a , float2 b , float3 c , float2 vertex);
bool InCircle ( Circle circle , float2 vertex);

Circle SmallerCircle(float2 a, float2 b);
Circle SmallerCircle(float2 a, float2 b , float2 c);

#ifdef USE_CIRCLE_TEST
#define InGenFaceTest(face,vertex) InCircle(face.boundary,vertex)
#else
#ifdef USE_BOUNDARY
#define InGenFaceTest(face,vertex) ((vertex.x >= face.Min.x && vertex.x <= face.Max.x) && \
(vertex.y >= face.Min.y && vertex.y <= face.Max.y))
#else
#define InGenFaceTest(face,vertex) false
#endif
#endif


// ------------ CreateTriangle  ------------ //

// create triangle(link the he) base on 3 vertex
__device__
void CreateTriangle(const DATA_TYPE *VertexList, 
                    HalfEdge *HEList,
                    ThreadInfo *threadInfo,
                    const uint Vert1ID, const uint Vert2ID, const uint Vert3ID,
				    HalfEdge  *he1, uint he1ID,
					HalfEdge  *he2, uint he2ID,
					HalfEdge  *he3, uint he3ID,
					const bool saveHE , bool *ccw)
{
	// create 3 new he
	he1ID = InitNewHalfEdge(Vert1ID, he1, threadInfo);
	he2ID = InitNewHalfEdge(Vert2ID, he2, threadInfo);
	he3ID = InitNewHalfEdge(Vert3ID, he3, threadInfo);

	// read the 3 points
	DATA_TYPE vert1 = VertexList[Vert1ID];
	DATA_TYPE vert2 = VertexList[Vert2ID];
	DATA_TYPE vert3 = VertexList[Vert3ID];

	ccw =  ( ( vert2.x - vert1.x ) * ( vert3.y - vert1.y ) - ( vert3.x - vert1.x ) * ( vert2.y - vert1.y ) ) > 0;
	
	// ccw test
	if ( ccw ) {
		he1->nextEdgeID = he2ID;
		he2->nextEdgeID = he3ID;
		he3->nextEdgeID = he1ID;
	} else {
		he1->nextEdgeID = he3ID;
		he3->nextEdgeID = he2ID;
		he2->nextEdgeID = he1ID;
	}
	
	if(saveHE){
		// update the new he in the memory
        HEList[he1ID] = he1;
        HEList[he2ID] = he2;
        HEList[he3ID] = he3;
	}
}

#endif /* H_TRIANGLE_UTILS */


