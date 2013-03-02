#include "includes.h"

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
template<bool save>  __device__
void CreateTriangle( const DATA_TYPE *VertexList, 
                     HalfEdge    *HEList,
                     ThreadInfo *threadInfo,
                     const uint Vert1ID, const uint Vert2ID, const uint Vert3ID,
                     HalfEdge  *he1,
                     HalfEdge  *he2,
                     HalfEdge  *he3,
                     uint3     *heID,
                     bool      *ccw)
{
    
    
    // read the 3 points
    DATA_TYPE vert1 = VertexList[Vert1ID];
    DATA_TYPE vert2 = VertexList[Vert2ID];
    DATA_TYPE vert3 = VertexList[Vert3ID];

    *ccw =  (( ( vert2.x - vert1.x ) * ( vert3.y - vert1.y ) - ( vert3.x - vert1.x ) * ( vert2.y - vert1.y ) ) > 0);
 
    // create 3 new he
    heID->x = InitNewHalfEdge(Vert1ID, he1, threadInfo);
    heID->y = InitNewHalfEdge(Vert2ID, he2, threadInfo);
    heID->z = InitNewHalfEdge(Vert3ID, he3, threadInfo);

    // ccw test
    if ( *ccw ) {
        he1->nextEdgeID = heID->y;
        he2->nextEdgeID = heID->z;
        he3->nextEdgeID = heID->x;
    } else {
        he1->nextEdgeID = heID->z;
        he3->nextEdgeID = heID->y;
        he2->nextEdgeID = heID->x;
    }
    
    if(save){
        // update the new he in the memory
        HEList[heID->x] = *he1;
        HEList[heID->y] = *he2;
        HEList[heID->z] = *he3;
    }
}

#endif /* H_TRIANGLE_UTILS */


