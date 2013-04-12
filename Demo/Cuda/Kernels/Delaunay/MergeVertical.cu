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
#include "..\Utils\cutil_math.h"

#define uint unsigned int
#define DATA_TYPE float2

#include "includes.h"
#include "IOBuffer.h"
#include "HalfEdgeUtils.h"
#include "FaceUtils.h"
#include "TriangleUtils.h"
#include "BoundaryUtils.h"

typedef struct {
    
// num of the threads
    uint ThreadNum;

// num of the max number of elements in stack
    uint stackMaxSize;
    
// Num of current depth
    uint depth;
    
// Num of thread per row
    uint ThreadNumPerRow;
    
// Num of thread per row
    uint HorizontalThreadNum;
    
} MV_threadParam;


//////////////////////////////////////////////////////////////////////////////////
// Main 
//////////////////////////////////////////////////////////////////////////////////

extern "C" __global__ 
void merge(const DATA_TYPE   *VertexList,
           HalfEdge          *HEList,
           BoundaryNode      *BoundaryList,
           Face              *FaceList,
           DelaunayNode      *stack,
           uint              *UintStack,
           ThreadInfo        *threadInfoArray,
           const MV_threadParam params)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    
    uint threadID_X = (2*x + 1)*pow( 2.0f, (int)params.depth) - 1;
    uint threadID = threadID_X + params.ThreadNumPerRow*y;
    
    if(threadID   >= params.ThreadNum			|| 
       y          >= params.HorizontalThreadNum || 
       threadID_X >= params.ThreadNumPerRow)
        return;
    
    // get region sides that we are going to merge
    uint leftSideID = threadID_X + (params.ThreadNumPerRow + 1)*y;
    uint rightSideID = leftSideID + 1;
    
    MergeVInfo mergeVInfo;
    mergeVInfo.threadID = threadID;
    
    mergeVInfo.LeftHEStack.offset = (uint)(2 * threadID * STACK_MAX_NUM);
    mergeVInfo.RightHEStack.offset = (uint)((2 * threadID + 1) * STACK_MAX_NUM);
    
    // get the left/right regions thread
    ThreadInfo leftThreadInfo = threadInfoArray[leftSideID];
    ThreadInfo rightThreadInfo = threadInfoArray[rightSideID];
    
    // clean the stack for the merging stack
    StackReset(&mergeVInfo.LeftHEStack);
    StackReset(&mergeVInfo.RightHEStack);
    
    // the low and high boundary nodes
    uint leftTmpNodeID, rightTmpNodeID;
    BoundaryNode leftTmpNode, rightTmpNode;
    
    uint leftPrevTmpNodeID, rightNextTmpNodeID;
    BoundaryNode leftPrevTmpNode, rightNextTmpNode;


    // ============================================================== Init
    
    // get the low and high boundary nodes
    leftTmpNodeID = leftThreadInfo.Boundary_Y_MaxID;
    rightTmpNodeID = rightThreadInfo.Boundary_Y_MaxID;
    
    // load the boundary nodes
    leftTmpNode  = BoundaryList[leftTmpNodeID];
    rightTmpNode = BoundaryList[rightTmpNodeID];
    //rightTmpNodeID = rightTmpNode.PrevNodeID;
    //rightTmpNode = GetBoundaryNode(rightTmpNodeID, rightThreadInfo);
    
    leftPrevTmpNodeID = leftTmpNode.PrevNodeID;
    rightNextTmpNodeID = rightTmpNode.NextNodeID;
    
    leftPrevTmpNode = BoundaryList[leftPrevTmpNodeID];
    rightNextTmpNode = BoundaryList[rightNextTmpNodeID];
    
    // get the edge of each side
    HalfEdge leftHE = HEList[leftTmpNode.halfEdgeID];
    HalfEdge rightHE = HEList[rightTmpNode.halfEdgeID];
    
    // get the vertex
    float2 vert1 =VertexList[leftHE.startVertexID];
    float2 vert2 =VertexList[rightHE.startVertexID];
    
    // get the prev edge of each side
    HalfEdge leftPrevHE = HEList[leftPrevTmpNode.halfEdgeID];
    HalfEdge rightNextHE = HEList[rightNextTmpNode.halfEdgeID];
    
    // get the vertex
    float2 leftPrevVert = VertexList[leftPrevHE.startVertexID];
    float2 rightNextVert = VertexList[rightNextHE.startVertexID];

    
    
    bool leftTest = SideTest(leftPrevVert, vert1, vert2);
    bool rightTest = SideTest(vert2, rightNextVert, vert1);
    
    // loop the end of the boundares
    // check the vert1 and vert 2 if they are valid ?
    while (!(leftTest && rightTest)
            && leftTmpNodeID != leftThreadInfo.Boundary_Y_MinID
            && rightTmpNodeID != rightThreadInfo.Boundary_Y_MinID)
    {
        // select witch one to drop
        if (!leftTest){
            // move to the prev left BN node 
            leftTmpNodeID = leftPrevTmpNodeID;
            leftTmpNode = leftPrevTmpNode;
            
            leftPrevTmpNodeID = leftPrevTmpNode.PrevNodeID;
            leftPrevTmpNode = BoundaryList[leftPrevTmpNodeID];
            
            // reset left he
            leftHE = leftPrevHE;
            leftPrevHE = HEList[leftPrevTmpNode.halfEdgeID];
            
            // update the vertix
            vert1 = VertexList[leftHE.startVertexID];
            leftPrevVert = VertexList[leftPrevHE.startVertexID];
        }
        
        if (!rightTest){
            
            // move to the next right node
            rightTmpNodeID = rightNextTmpNodeID;
            rightTmpNode = rightNextTmpNode;
            
            rightNextTmpNodeID = rightNextTmpNode.NextNodeID;
            rightNextTmpNode = BoundaryList[rightNextTmpNodeID];
            
            // reset the right he
            rightHE = rightNextHE;
            rightNextHE = HEList[rightNextTmpNode.halfEdgeID];
            
            // update right vertex
            vert2 = VertexList[rightHE.startVertexID];
            rightNextVert = VertexList[rightNextHE.startVertexID];
        }
        
        leftTest = SideTest(leftPrevVert, vert1, vert2);
        rightTest = SideTest(vert2, rightNextVert, vert1);
    }
    
    // ============================================================== Init
    
    ///// push the boundary 
    
    
    // push all the left he of the boundary
    {
        uint tmpNodeID = leftThreadInfo.Boundary_Y_MinID;
        BoundaryNode tmpNode = BoundaryList[tmpNodeID];
        
        // push all the left he of the boundary
        while (tmpNodeID != leftTmpNodeID){
            // push the he 
            StackPush(UintStack, tmpNode.halfEdgeID, &mergeVInfo.LeftHEStack);
            
            // move to the next bn
            tmpNodeID = tmpNode.NextNodeID;
            tmpNode = BoundaryList[tmpNodeID];
        }
        
        if (mergeVInfo.LeftHEStack.end == 0){ 
            // push the he 
            StackPush(UintStack, tmpNode.halfEdgeID, &mergeVInfo.LeftHEStack);
        }
    }
    
    // push all the right he of the boundary
    {
        
        uint tmpNodeID = rightThreadInfo.Boundary_Y_MinID;
        BoundaryNode tmpNode = BoundaryList[tmpNodeID];
        tmpNodeID = tmpNode.PrevNodeID;
        tmpNode = BoundaryList[tmpNode.PrevNodeID];
        
        while (tmpNodeID != rightTmpNodeID)
        {
            // push the he 
            StackPush(UintStack, tmpNode.halfEdgeID, &mergeVInfo.RightHEStack);
            
            // move to the prev bn
            tmpNodeID = tmpNode.PrevNodeID;
            tmpNode = BoundaryList[tmpNodeID];
        }
        
        // push and the last node
        StackPush(UintStack, tmpNode.halfEdgeID, &mergeVInfo.RightHEStack);
    }

    // ============================================================== Merging Verticals
    //// start the merging 
    
    // init the he
    uint leftHE_ID, rightHE_ID;
    StackPull(UintStack, &leftHE_ID, &mergeVInfo.LeftHEStack);
    StackPull(UintStack, &rightHE_ID, &mergeVInfo.RightHEStack);
    
    leftHE = HEList[leftHE_ID];
    rightHE = HEList[rightHE_ID];
    
    // get the vertex of the base line
    uint rightBaseVecId = rightHE.startVertexID;
    float2 rightBaseVec = VertexList[rightBaseVecId];
    uint leftBaseVecId = HEList[leftHE.nextEdgeID].startVertexID;
    float2 leftBaseVec = VertexList[leftBaseVecId];
    float2 testVecA, testVecB;
    
    uint prevHE_ID = UNSET;
    HalfEdge prevHE;
    bool rightFinish = false, leftFinish = false;
    bool first = true;
    
    while (true)
    {
        float2 leftHE_Vec, rightHE_Vec;
        uint leftHE_VecID, rightHE_VecID;
        uint thirdVecId;
        bool right;
    
        leftHE = HEList[leftHE_ID];
        rightHE = HEList[rightHE_ID];
    
        ///++ #region Select the base vec
        
        
        // get the vertex of the base line
        if (rightFinish)
        {
            if (rightHE.nextEdgeID != UNSET)
            {
                HalfEdge rightHENext = HEList[rightHE.nextEdgeID];
                rightBaseVec = VertexList[rightHENext.startVertexID];
                rightBaseVecId = rightHENext.startVertexID;
            }
            else
            {
                HalfEdge rightTwin = HEList[rightHE.twinEdgeID];
                rightBaseVec = VertexList[rightTwin.startVertexID];
                rightBaseVecId = rightTwin.startVertexID;
            }
        }
        else
        {
            if (rightHE.nextEdgeID != UNSET)
            {
                rightBaseVecId = rightHE.startVertexID;
                rightBaseVec = VertexList[rightBaseVecId];
            }
            else
            {
                rightBaseVec = VertexList[rightHE.startVertexID];
                rightBaseVecId = rightHE.startVertexID;
            }
        }
        
        if (leftFinish)
        {
            if (leftHE.nextEdgeID != UNSET)
            {
                leftBaseVec = VertexList[leftHE.startVertexID];
                leftBaseVecId = leftHE.startVertexID;
            }
            else
            {
                HalfEdge leftTwin = HEList[leftHE.twinEdgeID];
                leftBaseVec = VertexList[leftTwin.startVertexID];
                leftBaseVecId = leftTwin.startVertexID;
            }
        }
        else
        {
            if (leftHE.nextEdgeID != UNSET)
            {
                leftBaseVecId = HEList[leftHE.nextEdgeID].startVertexID;
                leftBaseVec = VertexList[leftBaseVecId];
            }
            else
            {
                HalfEdge leftTwin = HEList[leftHE.twinEdgeID];
                leftBaseVec = VertexList[leftTwin.startVertexID];
                leftBaseVecId = leftTwin.startVertexID;
            }
        }
        ///-- #endregion
    
        
        
        
        
        
        
        
        
        // left side processing
        ///++ #region left side processing
        {
            if (leftHE.nextEdgeID != UNSET)
            {
                HalfEdge leftHENext     = HEList[leftHE.nextEdgeID];
                HalfEdge leftHENextNext = HEList[leftHENext.nextEdgeID];
                
                testVecA        = VertexList[leftHE.startVertexID];
                testVecB        = VertexList[leftHENextNext.startVertexID];
                
                leftHE_Vec      = testVecA;
                leftHE_VecID    = leftHE.startVertexID;
                
                // left side process
                while (    SideTest(leftBaseVec, rightBaseVec, testVecB) 
                        && InCircleTest(rightBaseVec, leftBaseVec, testVecA, testVecB)// if the next test face is not extist break the loop
                        && leftHENext.twinEdgeID != UNSET)
                {
                    // push the twin of the the leftHENextNext
                    if (leftHENextNext.twinEdgeID != UNSET) // TODO: fix this problem
                        StackPush(UintStack, leftHENextNext.twinEdgeID,  &mergeVInfo.LeftHEStack);
                    else
                    {
                        HalfEdge dummyHE;
                        uint dummyHEID = InitNewHalfEdge(leftHE.startVertexID, &dummyHE,  &leftThreadInfo);
                        dummyHE.twinEdgeID = leftHENext.nextEdgeID;
                        HEList[dummyHEID]=dummyHE;
                        
                        StackPush(UintStack, dummyHEID,  &mergeVInfo.LeftHEStack);
                    }
                    
                    // remove the triangle of the corresponded HE 
                    Face face       = GetFace(FaceList, leftHE.faceID);
                    face.halfEdgeID =  UNSET;
                    SetFace(FaceList, face, leftHE.faceID);
                    
                    // move to the next test face
                    leftHE_ID       = leftHENext.twinEdgeID;
                    leftHE          = HEList[leftHE_ID];
                    
                    // load and the HE of the next face
                    leftHENext      = HEList[leftHE.nextEdgeID];
                    leftHENextNext  = HEList[leftHENext.nextEdgeID];
                    
                    // load the new test vecAB
                    testVecA        = VertexList[leftHE.startVertexID];
                    testVecB        = VertexList[leftHENextNext.startVertexID];
                    
                    leftHE_Vec      = testVecA;
                    leftHE_VecID    = leftHE.startVertexID;
                }
            }
            else
            {
                HalfEdge leftTwin   = HEList[leftHE.twinEdgeID];
                leftHE_Vec          = VertexList[leftTwin.startVertexID];
                leftHE_VecID        = leftTwin.startVertexID;
            }
        }
        ///-- #endregion
    
        
        
        
        
        
        
        
        
        
        
        
        // right side processing
        ///++ #region right side processing
        {
            if (rightHE.nextEdgeID != UNSET)
            {
                HalfEdge rightHENext        = HEList[rightHE.nextEdgeID];
                HalfEdge rightHENextNext    = HEList[rightHENext.nextEdgeID];
                
                testVecA        = VertexList[rightHENext.startVertexID];
                testVecB        = VertexList[rightHENextNext.startVertexID];
                
                rightHE_Vec     = testVecA;
                rightHE_VecID   = rightHENext.startVertexID;
                
                // left side process
                while (    SideTest(leftBaseVec, rightBaseVec, testVecB) 
                        && InCircleTest(rightBaseVec, leftBaseVec, testVecA, testVecB)
                        && rightHENextNext.twinEdgeID != UNSET)
                {
                    // push the twin of the the rightHENext
                    if (rightHENext.twinEdgeID != UNSET) // TODO: fix this problem
                        StackPush(UintStack, rightHENext.twinEdgeID,  &mergeVInfo.RightHEStack);
                    else
                    {
                        HalfEdge dummyHE;
                        uint dummyHEID      = InitNewHalfEdge(rightHENextNext.startVertexID, &dummyHE,  &rightThreadInfo);
                        dummyHE.twinEdgeID  = rightHE.nextEdgeID;
                        HEList[dummyHEID]   = dummyHE;
                        
                        StackPush(UintStack, dummyHEID, &mergeVInfo.RightHEStack);
                    }
                    
                    // remove the triangle of the corresponded HE 
                    Face face       = GetFace(FaceList, rightHE.faceID);
                    face.halfEdgeID =  UNSET;
                    SetFace(FaceList, face, rightHE.faceID);
                    
                    // move to the next test face
                    rightHE_ID      = rightHENextNext.twinEdgeID;
                    rightHE         = HEList[rightHE_ID];
                    
                    // load and the HE of the next face
                    rightHENext     = HEList[rightHE.nextEdgeID];
                    rightHENextNext = HEList[rightHENext.nextEdgeID];
                    
                    // load the new test vecAB
                    testVecA        = VertexList[rightHENext.startVertexID];
                    testVecB        = VertexList[rightHENextNext.startVertexID];
                    
                    rightHE_Vec     = testVecA;
                    rightHE_VecID   = rightHENext.startVertexID;
                }
            }
            else
            {
                HalfEdge rightTwin  = HEList[rightHE.twinEdgeID];
                rightHE_Vec         = VertexList[rightTwin.startVertexID];
                rightHE_VecID       = rightTwin.startVertexID;
            }
        }
        ///-- #endregion
    
        
        
        
        
        
        
        // choice the best side connection
        ///++ #region third vector selection
        
        bool rightSideOk    = SideTest(leftBaseVec, rightBaseVec, rightHE_Vec);
        bool leftSideOk     = SideTest(leftBaseVec, rightBaseVec, leftHE_Vec);
        
        bool forceSide = !(rightSideOk && leftSideOk);
        
        if (rightFinish || leftFinish)
        {
            if (rightFinish)
            {
                thirdVecId = leftHE_VecID;
                if (leftSideOk)
                    right = false;
                else
                    break;
            }
            else
            {
                thirdVecId = rightHE_VecID;
                if (rightSideOk)
                    right = true;
                else
                    break;
            }
        }else if (forceSide)
        {
            if (!rightSideOk)
            {
                thirdVecId = leftHE_VecID;
                right = false;
            }
            else
            {
                thirdVecId = rightHE_VecID;
                right = true;
            }
        }
        else
        {
            if (InCircleTest(leftBaseVec, rightBaseVec, leftHE_Vec, rightHE_Vec))
            {
                thirdVecId = rightHE_VecID;
                right = true;
            }
            else
            {
                thirdVecId = leftHE_VecID;
                right = false;
            }
        }
        
        ///-- #endregion
    
        
        
        
        
        
        ///++ #region Create the Face
        
        {
            // create 3 new he
            HalfEdge he1, he2, he3;
            uint he1ID, he2ID, he3ID;
            bool ccw;
            // don't store the results
            CreateTriangle<false>(VertexList, HEList, &leftThreadInfo, 
                                  rightBaseVecId, leftBaseVecId, thirdVecId,
                                  &he1, &he2, &he3,
                                  &he1ID, &he2ID, &he3ID,
                                  &ccw);
            
            // link the new face
            if (right)
            {
                
                // link the he3
                if (rightHE.nextEdgeID != UNSET)
                {
                    he3.twinEdgeID = rightHE_ID;
                    rightHE.twinEdgeID = he3ID;
                }
                else 
                {
                    he3ID = rightHE.twinEdgeID;
                    
                    // ccw test
                    if (ccw)
                    {
                        he2.nextEdgeID = he3ID;
                    }
                    else
                    {
                        he1.nextEdgeID = he3ID;
                    }
                    
                }
                
                // link the he1 with the prev he
                if (prevHE_ID != UNSET)
                {
                    prevHE              = HEList[prevHE_ID];
                    
                    prevHE.twinEdgeID   = he1ID;
                    he1.twinEdgeID      = prevHE_ID;
                    
                    // save the new prev value
                    HEList[prevHE_ID]   = prevHE;
                }
                
                // set the new prev HE
                prevHE_ID   = he2ID;
                prevHE      = he2;
                
            }
            else
            {
                // link the he2 
                if (leftHE.nextEdgeID != UNSET)
                {
                    he2.twinEdgeID      = leftHE_ID;
                    leftHE.twinEdgeID   = he2ID;
                }
                else
                {
                    he2ID = leftHE.twinEdgeID;
                    
                    // ccw test
                    if (ccw)
                    {
                        he1.nextEdgeID = he2ID;
                    }
                    else
                    {
                        he3.nextEdgeID = he2ID;
                    }
                }
                
                // link the he1 with the prev he
                if (prevHE_ID != UNSET)
                {
                    prevHE              = HEList[prevHE_ID];
                    
                    prevHE.twinEdgeID   = he1ID;
                    he1.twinEdgeID      = prevHE_ID;
                    
                    // save the new prev value
                    HEList[prevHE_ID]   =prevHE;
                }
                
                // set the new prev HE
                prevHE_ID   = he3ID;
                prevHE      = he3;
                
    
            }
            
            // the first time store the first merging
            if(first){
                // set the last HE_ID 
                leftThreadInfo.LeftFirstMergingHEID = he1ID;
                rightThreadInfo.RightFirstMergingHEID = he1ID;
                first = false;
            }		
            
    
            // create a new face
            CreateFace(HEList,
                       FaceList,
                       &he1, he1ID,
                       &he2,
                       &he3,
                       &leftThreadInfo);
            
            // store the results
            HEList[he1ID] = he1;
            HEList[he2ID] = he2;
            HEList[he3ID] = he3;
            
            HEList[rightHE_ID] = rightHE;
            HEList[leftHE_ID]  = leftHE;
            
        }
        
        ///-- #endregion
    
        
        
        
        // pull the next he
        if (right && !rightFinish)
        {
            uint tmpSave = rightHE_ID;
            if (!StackPull(UintStack, &rightHE_ID, &mergeVInfo.RightHEStack))
            {
                rightHE_ID  = tmpSave;
                rightFinish = true;
            }
        }
        
        if (!right && !leftFinish)
        {
            uint tmpSave = leftHE_ID;
            if (!StackPull(UintStack, &leftHE_ID, &mergeVInfo.LeftHEStack))
            {
                leftHE_ID  = tmpSave;
                leftFinish = true;
            }
        }
        
        if (rightFinish && leftFinish)
            break;
    
    }
    
    // set the last HE_ID 
    leftThreadInfo.LeftLastMergingHEID   = prevHE_ID;
    rightThreadInfo.RightLastMergingHEID = prevHE_ID;
    
    
    // ============================================================== Finalize
    // save the merge info
    threadInfoArray[leftThreadInfo.threadID]  = leftThreadInfo;
    threadInfoArray[rightThreadInfo.threadID] = rightThreadInfo;
}


























