
#include "includes.h"

#ifndef H_FACE_UTILS
#define H_FACE_UTILS



// ------------ CreateFace  ------------ //

// create a face base on the exist halfEdge
__device__
uint2 CreateFace(HalfEdge   *HEList,
                 Face       *FaceList,
                 uint        HalfEdgeID,
                 ThreadInfo *threadInfo)
{
    uint2 newFaceID;
    uint heID;
    Face newFace;
    
    //set the face ID
    newFaceID = threadInfo->lastFaceID;

    // inc the last id for faces
    threadInfo->lastFaceID.x++;
    
    // get the half edges and link them with the face
    HalfEdge he = HEList[HalfEdgeID];		                //  get the first he
    he.faceID 	= newFaceID;								//  link the he with the face
    HEList[HalfEdgeID] = he;				                //  set back the face 
    
    heID 		= he.nextEdgeID;
    he 			= HEList[heID];                  			//  move to the next he
    he.faceID 	= newFaceID;								//  link the he with the face
    HEList[heID] = he;				                        //  set back the face

    heID 		= he.nextEdgeID;
    he 			= HEList[heID];                  			//  move to the next he
    he.faceID 	= newFaceID;								//  link the he with the face
    HEList[heID] = he;				                        //  set back the face


    // link the new face to the he
    newFace.halfEdgeID = HalfEdgeID;
    
    // write the face back to the memory
    SetFace(FaceList, newFace , newFaceID);
    
    // return the face ID
    return newFaceID;
}

// create a face base on the exist halfEdge
__device__
uint2 CreateFace(HalfEdge   *HEList,
                 Face       *FaceList,
                 HalfEdge   *he1, uint he1ID,
                 HalfEdge   *he2,
                 HalfEdge   *he3,
                 ThreadInfo *threadInfo)
{
    uint2 newFaceID;
    uint heID;
    Face newFace;
    
    //set the face ID
    newFaceID = threadInfo->lastFaceID;

    // inc the last id for faces
    threadInfo->lastFaceID.x++;
    
    // get the half edges and link them with the face
    he1->faceID 	= newFaceID;								//  link the he1 with the face
    he2->faceID 	= newFaceID;								//  link the he2 with the face
    he3->faceID 	= newFaceID;								//  link the he3 with the face

    // link the new face to the he
    newFace.halfEdgeID = he1ID;
    
    // write the face back to the memory
    SetFace(FaceList, newFace , newFaceID);
    
    // return the face ID
    return newFaceID;
}
#endif /* H_FACE_UTILS */





