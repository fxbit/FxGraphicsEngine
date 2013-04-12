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
                     uint      *he1ID,
                     uint      *he2ID,
                     uint      *he3ID,
                     bool      *ccw)
{
    // read the 3 points
    DATA_TYPE vert1 = VertexList[Vert1ID];
    DATA_TYPE vert2 = VertexList[Vert2ID];
    DATA_TYPE vert3 = VertexList[Vert3ID];

    *ccw =  (( ( vert2.x - vert1.x ) * ( vert3.y - vert1.y ) - ( vert3.x - vert1.x ) * ( vert2.y - vert1.y ) ) > 0);
 
    // create 3 new he
    *he1ID = InitNewHalfEdge(Vert1ID, he1, threadInfo);
    *he2ID = InitNewHalfEdge(Vert2ID, he2, threadInfo);
    *he3ID = InitNewHalfEdge(Vert3ID, he3, threadInfo);

    // ccw test
    if ( *ccw ) {
        he1->nextEdgeID = *he2ID;
        he2->nextEdgeID = *he3ID;
        he3->nextEdgeID = *he1ID;
    } else {
        he1->nextEdgeID = *he3ID;
        he3->nextEdgeID = *he2ID;
        he2->nextEdgeID = *he1ID;
    }
    
    if(save){
        // update the new he in the memory
        HEList[*he1ID] = *he1;
        HEList[*he2ID] = *he2;
        HEList[*he3ID] = *he3;
    }
}


// ------------ Side test ------------ //

/// Check if a point is in the right/left of a vector
__device__
bool SideTest(float2 orig , float2 A, float2 B)
{
    float2 OA = A - orig;
    float2 OB = B - orig;

    OA = normalize( OA );
    OB = normalize( OB );

    return (OA.x * OB.y - OA.y * OB.x) < -0.0000001;
}


// ------------ InCircle test ------------ //

// test if the vertex is inside of circle that generated from a triangle
__device__
bool InCircleTest ( float2 a , float2 b , float2 c , float2 vertex)
{
    // CCW test
    if ( ( ( b.x - a.x ) * ( c.y - a.y ) - ( c.x - a.x ) * ( b.y - a.y ) ) > 0 )  {
        //swap vertex
        float2 temp = b;
        b = c;
        c = temp;
    }
    
    // finv the viff in each axes
    float2 va = a-vertex;
    float2 vb = b-vertex;
    float2 vc = c-vertex;

    // calc the veterminal
    float abdet = va.x * vb.y - vb.x * va.y;
    float bcdet = vb.x * vc.y - vc.x * vb.y;
    float cadet = vc.x * va.y - va.x * vc.y;
    float alift = va.x * va.x + va.y * va.y;
    float blift = vb.x * vb.x + vb.y * vb.y;
    float clift = vc.x * vc.x + vc.y * vc.y;

    return (alift * bcdet + blift * cadet + clift * abdet < 0);
}

// create a face base on the exist halfEdge
__device__
void UpdateFace(Face    *FaceList,
                HalfEdge he1, uint he1ID)
{
    uint2 oldFaceID;
    Face oldFace;

    // get the face from the given half edge
    oldFace 	        = GetFace(FaceList, he1.faceID);
    oldFace.halfEdgeID  = he1ID;
    oldFaceID 	        = he1.faceID;
    
    // write the face back to the memory
    SetFace(FaceList, oldFace, oldFaceID);
}


// ------------ FixTheFaces ------------ //


// Fix all the delaunay nodes that exist in the stack
__device__
void FixStackFaces(const DATA_TYPE  *VertexList, 
                   HalfEdge         *HEList,
                   Face             *FaceList,
                   DelaunayNode     *stack,
                   ThreadInfo       *threadInfo)
{
    int g=0;
    // fix the faces
    while(g<MAX_FACE_CORRECTIONS)
    {
        DelaunayNode node;
        bool testPull = PullDelaunayNode(stack, &node, threadInfo);
        
        // pull the first node
        if( testPull ){
            
            // get the face that is under test
            uint he_a1ID = node.RootHalfEdgeID;
            HalfEdge he_a1 = HEList[he_a1ID];
            float2 he_a1v = VertexList[he_a1.startVertexID];
            
            uint he_a2ID = he_a1.nextEdgeID;
            HalfEdge he_a2 = HEList[he_a2ID];
            float2 he_a2v = VertexList[he_a2.startVertexID];
            
            uint he_a3ID = he_a2.nextEdgeID;
            HalfEdge he_a3 = HEList[he_a3ID];
            float2 he_a3v = VertexList[he_a3.startVertexID];
            
            
            // get the oposide point 
            uint he_b1ID = node.NextFaceHalfEdgeID;
            HalfEdge he_b1 = HEList[he_b1ID];
            float2 he_b1v = VertexList[he_b1.startVertexID];
            
            uint he_b2ID = he_b1.nextEdgeID;
            HalfEdge he_b2 = HEList[he_b2ID];
            float2 he_b2v = VertexList[he_b2.startVertexID];
            
            uint he_b3ID = he_b2.nextEdgeID;
            HalfEdge he_b3 = HEList[he_b3ID];
            float2 he_b3v = VertexList[he_b3.startVertexID];
            
            
            bool test = InCircleTest(he_a1v, he_a2v, he_a3v, he_b3v);
            
            // test if the face need correction
            if(test){
                
                // copy the start vertex
                he_a2.startVertexID = he_b3.startVertexID;
                he_b1.startVertexID = he_a1.startVertexID;
                
                // change the next edge
                he_a1.nextEdgeID 		= he_b2ID;
                he_b2.nextEdgeID 		= he_a2ID;
                he_a2.nextEdgeID 	    = he_a1ID;

                he_b3.nextEdgeID 		= he_a3ID;
                he_a3.nextEdgeID 		= he_b1ID;
                he_b1.nextEdgeID 	    = he_b3ID;
                
                // update face that each edge exist
                he_b2.faceID = he_a1.faceID;
                he_a2.faceID = he_a1.faceID;

                he_a3.faceID = he_b3.faceID;
                he_b1.faceID = he_b3.faceID;

                // store the new half edge
                HEList[he_a1ID]=he_a1;
                HEList[he_a2ID]=he_a2;
                HEList[he_a3ID]=he_a3;
                HEList[he_b1ID]=he_b1;
                HEList[he_b2ID]=he_b2;
                HEList[he_b3ID]=he_b3;
                

                // update the faces
                UpdateFace(FaceList,
                           he_a1, he_a1ID);
                
                UpdateFace(FaceList,
                           he_b1, 	he_b1ID);
                
                // add to the stack and the 2 new twin faces
                // push the oposide of he_b1
                if( he_b3.twinEdgeID != UNSET )
                {
                    DelaunayNode newNode;
                    newNode.RootHalfEdgeID 	= he_b1ID;
                    newNode.NextFaceHalfEdgeID = he_b3.twinEdgeID;
                    
                    PushDelaunayNode( stack, newNode , threadInfo );
                }
                
                // push the oposide of he_b1
                if( he_b2.twinEdgeID != UNSET )
                {
                    DelaunayNode newNode;
                    newNode.RootHalfEdgeID 	= he_a1ID;
                    newNode.NextFaceHalfEdgeID = he_b2.twinEdgeID;
                    
                    PushDelaunayNode( stack, newNode , threadInfo );
                }
                    
            }
        }else
            break;
        
        g++;
    }

}



#endif /* H_TRIANGLE_UTILS */


