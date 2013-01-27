
#include "mv_includes.hlslh"

//////////////////////////////////////////////////////////////////////////////////
// Get starting Node 
//		-> fix the starting nodes
//////////////////////////////////////////////////////////////////////////////////

void GetStartingNode( ThreadInfo leftThreadInfo, 		  ThreadInfo rightThreadInfo,
				  inout uint leftTmpNodeID,  	inout BoundaryNode leftTmpNode,
				  inout uint rightTmpNodeID, 	inout BoundaryNode rightTmpNode,
				  inout uint leftPrevTmpNodeID, inout BoundaryNode leftPrevTmpNode,
				  inout uint rightNextTmpNodeID,inout BoundaryNode rightNextTmpNode)
{
	// get the low and high boundary nodes
	leftTmpNodeID = leftThreadInfo.Boundary_Y_MaxID;
	rightTmpNodeID = rightThreadInfo.Boundary_Y_MaxID;
	
	// load the boundary nodes
	leftTmpNode = GetBoundaryNode(leftTmpNodeID, leftThreadInfo);
	rightTmpNode = GetBoundaryNode(rightTmpNodeID, rightThreadInfo);
	//rightTmpNodeID = rightTmpNode.PrevNodeID;
	//rightTmpNode = GetBoundaryNode(rightTmpNodeID, rightThreadInfo);
	
	leftPrevTmpNodeID = leftTmpNode.PrevNodeID;
	rightNextTmpNodeID = rightTmpNode.NextNodeID;
	
	leftPrevTmpNode = GetBoundaryNode(leftPrevTmpNodeID, leftThreadInfo);
	rightNextTmpNode = GetBoundaryNode(rightNextTmpNodeID, rightThreadInfo);
	
	// get the edge of each side
	HalfEdge leftHE = GetHalfEdge(leftTmpNode.halfEdgeID, leftThreadInfo);
	HalfEdge rightHE = GetHalfEdge(rightTmpNode.halfEdgeID, rightThreadInfo);
	
	HalfEdge leftPrevHE = GetHalfEdge(leftPrevTmpNode.halfEdgeID, leftThreadInfo);
	HalfEdge rightNextHE = GetHalfEdge(rightNextTmpNode.halfEdgeID, rightThreadInfo);
	
	// get the vertex
	float2 vert1 = GetVertex(leftHE.startVertexID, leftThreadInfo);
	float2 vert2 = GetVertex(rightHE.startVertexID, rightThreadInfo);
	
	float2 leftPrevVert = GetVertex(leftPrevHE.startVertexID, leftThreadInfo);
	float2 rightNextVert = GetVertex(rightNextHE.startVertexID, rightThreadInfo);

	
	
	bool leftTest = SideTest(leftPrevVert, vert1, vert2);
	bool rightTest = SideTest(vert2, rightNextVert, vert1);
	
	// loop the end of the boundares
	// check the vert1 and vert 2 if they are valid ?
	[allow_uav_condition] [loop]
	while (!(leftTest && rightTest)
			&& leftTmpNodeID != leftThreadInfo.Boundary_Y_MinID
			&& rightTmpNodeID != rightThreadInfo.Boundary_Y_MinID)
	{
		// select witch one to drop
		[branch]
		if (!leftTest)
		{
			// move to the prev left BN node 
			leftTmpNodeID = leftPrevTmpNodeID;
			leftTmpNode = leftPrevTmpNode;
	
			leftPrevTmpNodeID = leftPrevTmpNode.PrevNodeID;
			leftPrevTmpNode = GetBoundaryNode(leftPrevTmpNodeID, leftThreadInfo);
	
			// reset left he
			leftHE = leftPrevHE;
			leftPrevHE = GetHalfEdge(leftPrevTmpNode.halfEdgeID, leftThreadInfo);
	
			// update the vertix
			vert1 = GetVertex(leftHE.startVertexID, leftThreadInfo);
			leftPrevVert = GetVertex(leftPrevHE.startVertexID, leftThreadInfo);
		}
		
		if (!rightTest)
		{
	
			// move to the next right node
			rightTmpNodeID = rightNextTmpNodeID;
			rightTmpNode = rightNextTmpNode;
	
			rightNextTmpNodeID = rightNextTmpNode.NextNodeID;
			rightNextTmpNode = GetBoundaryNode(rightNextTmpNodeID, rightThreadInfo);
	
			// reset the right he
			rightHE = rightNextHE;
			rightNextHE = GetHalfEdge(rightNextTmpNode.halfEdgeID, rightThreadInfo);
	
			// update right vertex
			vert2 = GetVertex(rightHE.startVertexID, rightThreadInfo);
			rightNextVert = GetVertex(rightNextHE.startVertexID, rightThreadInfo);
		}
		
		leftTest = SideTest(leftPrevVert, vert1, vert2);
		rightTest = SideTest(vert2, rightNextVert, vert1);
	}

}


//////////////////////////////////////////////////////////////////////////////////
// Main 
//////////////////////////////////////////////////////////////////////////////////


[numthreads(MERGE_VERTICAL_X, MERGE_VERTICAL_Y, 1)]
void main(uint3 threadId : SV_DispatchThreadID,
		  uint GI : SV_GroupIndex )
{
	if(threadId.y >= HorizontalThreadNum)
		return;
	
	uint threadID_X = (2*threadId.x + 1)*pow(2,depth) - 1;
		
	if(threadID_X >= ThreadNumPerRow)
		return;
	
	uint threadID = threadID_X + ThreadNumPerRow*threadId.y;
	
	if(threadID >= ThreadNum)
		return;
	
	// get region sides that we are going to merge
	uint leftSideID = threadID_X + (ThreadNumPerRow + 1)*threadId.y;
	uint rightSideID = threadID_X + 1 + (ThreadNumPerRow + 1)*threadId.y;
	
	MergeVInfo mergeVInfo;// = GetMergeVInfo(threadID);
	mergeVInfo.threadID = threadID;
	
	
#ifdef USE_SHARED_MEM
	mergeVInfo.LeftHEStack.offset = (uint)(2 * GI * stackMaxSize);
	mergeVInfo.RightHEStack.offset = (uint)((2 * GI + 1) * stackMaxSize);
#else
	mergeVInfo.LeftHEStack.offset = (uint)(2 * threadID * STACK_MAX_NUM);
	mergeVInfo.RightHEStack.offset = (uint)((2 * threadID + 1) * STACK_MAX_NUM);
#endif

	// get the left/right regions thread
	ThreadInfo leftThreadInfo = GetThreadInfo(leftSideID);
	ThreadInfo rightThreadInfo = GetThreadInfo(rightSideID);
	

	// clean the stack for the merging stack
	StackReset(mergeVInfo.LeftHEStack);
	StackReset(mergeVInfo.RightHEStack);
	
	// the low and high boundary nodes
	uint leftTmpNodeID, rightTmpNodeID;
	BoundaryNode leftTmpNode, rightTmpNode;
	
	uint leftPrevTmpNodeID, rightNextTmpNodeID;
	BoundaryNode leftPrevTmpNode, rightNextTmpNode;


	// ============================================================== Init
					
	// get the starting nodes
	GetStartingNode(leftThreadInfo, rightThreadInfo,
					leftTmpNodeID, leftTmpNode,
					rightTmpNodeID, rightTmpNode,
					leftPrevTmpNodeID, leftPrevTmpNode,
					rightNextTmpNodeID, rightNextTmpNode);


	///// push the boundary 
	
	// push all the left he of the boundary
	{
		uint tmpNodeID = leftThreadInfo.Boundary_Y_MinID;
		BoundaryNode tmpNode = GetBoundaryNode(tmpNodeID, leftThreadInfo);
	
		// push all the left he of the boundary
		[allow_uav_condition] [loop]
		while (tmpNodeID != leftTmpNodeID)
		{
			// push the he 
			StackPush(tmpNode.halfEdgeID, mergeVInfo.LeftHEStack);
	
			// move to the next bn
			tmpNodeID = tmpNode.NextNodeID;
			tmpNode = GetBoundaryNode(tmpNodeID, leftThreadInfo);
		}
	
		if (mergeVInfo.LeftHEStack.end == 0)
		{ 
			// push the he 
			StackPush(tmpNode.halfEdgeID, mergeVInfo.LeftHEStack);
		}
	}
	
	// push all the right he of the boundary
	{
	
		uint tmpNodeID = rightThreadInfo.Boundary_Y_MinID;
		BoundaryNode tmpNode = GetBoundaryNode(tmpNodeID, rightThreadInfo);
		tmpNodeID = tmpNode.PrevNodeID;
		tmpNode = GetBoundaryNode(tmpNode.PrevNodeID, rightThreadInfo);
	
		[allow_uav_condition] [loop]
		while (tmpNodeID != rightTmpNodeID)
		{
			// push the he 
			StackPush(tmpNode.halfEdgeID, mergeVInfo.RightHEStack);
	
			// move to the prev bn
			tmpNodeID = tmpNode.PrevNodeID;
			tmpNode = GetBoundaryNode(tmpNodeID, rightThreadInfo);
		}
	
		// push and the last node
		StackPush(tmpNode.halfEdgeID, mergeVInfo.RightHEStack);
	}

	// ============================================================== Merging Verticals
	//// start the merging 

	
	// init the he
	uint leftHE_ID, rightHE_ID;
	StackPull(leftHE_ID, mergeVInfo.LeftHEStack);
	StackPull(rightHE_ID, mergeVInfo.RightHEStack);
	
	HalfEdge leftHE = GetHalfEdge(leftHE_ID, leftThreadInfo);
	HalfEdge rightHE = GetHalfEdge(rightHE_ID, rightThreadInfo);
	
	// get the vertex of the base line
	uint rightBaseVecId = rightHE.startVertexID;
	float2 rightBaseVec = GetVertex(rightBaseVecId, rightThreadInfo);
	uint leftBaseVecId = GetHalfEdge(leftHE.nextEdgeID, leftThreadInfo).startVertexID;
	float2 leftBaseVec = GetVertex(leftBaseVecId, rightThreadInfo);
	float2 testVecA, testVecB;
	
	uint prevHE_ID = UNSET;
	HalfEdge prevHE;
	bool rightFinish = false, leftFinish = false;
	bool first = true;
	
	[allow_uav_condition] [loop]
	while (true)
	{
		float2 leftHE_Vec, rightHE_Vec;
		uint leftHE_VecID, rightHE_VecID;
		uint thirdVecId;
		bool right;

		leftHE = GetHalfEdge(leftHE_ID, leftThreadInfo);
		rightHE = GetHalfEdge(rightHE_ID, rightThreadInfo);

		
		
		
		
		
		
		///++ #region Select the base vec
		
		
		// get the vertex of the base line
		[branch]
		if (rightFinish)
		{
			[branch]
			if (rightHE.nextEdgeID != UNSET)
			{
				HalfEdge rightHENext = GetHalfEdge(rightHE.nextEdgeID, rightThreadInfo);
				rightBaseVec = GetVertex(rightHENext.startVertexID, rightThreadInfo);
				rightBaseVecId = rightHENext.startVertexID;
			}
			else
			{
				HalfEdge rightTwin = GetHalfEdge(rightHE.twinEdgeID, rightThreadInfo);
				rightBaseVec = GetVertex(rightTwin.startVertexID, rightThreadInfo);
				rightBaseVecId = rightTwin.startVertexID;
			}
		}
		else
		{
			[branch]
			if (rightHE.nextEdgeID != UNSET)
			{
				rightBaseVecId = rightHE.startVertexID;
				rightBaseVec = GetVertex(rightBaseVecId, rightThreadInfo);
			}
			else
			{
				rightBaseVec = GetVertex(rightHE.startVertexID, rightThreadInfo);
				rightBaseVecId = rightHE.startVertexID;
			}
		}
		
		[branch]
		if (leftFinish)
		{
			[branch]
			if (leftHE.nextEdgeID != UNSET)
			{
				leftBaseVec = GetVertex(leftHE.startVertexID, leftThreadInfo);
				leftBaseVecId = leftHE.startVertexID;
			}
			else
			{
				HalfEdge leftTwin = GetHalfEdge(leftHE.twinEdgeID, leftThreadInfo);
				leftBaseVec = GetVertex(leftTwin.startVertexID, leftThreadInfo);
				leftBaseVecId = leftTwin.startVertexID;
			}
		}
		else
		{
			[branch]
			if (leftHE.nextEdgeID != UNSET)
			{
				leftBaseVecId = GetHalfEdge(leftHE.nextEdgeID, leftThreadInfo).startVertexID;
				leftBaseVec = GetVertex(leftBaseVecId, rightThreadInfo);
			}
			else
			{
				HalfEdge leftTwin = GetHalfEdge(leftHE.twinEdgeID, leftThreadInfo);
				leftBaseVec = GetVertex(leftTwin.startVertexID, leftThreadInfo);
				leftBaseVecId = leftTwin.startVertexID;
			}
		}
		///-- #endregion

		
		
		
		
		
		
		
		
		// left side processing
		///++ #region left side processing
		{
			[branch]
			if (leftHE.nextEdgeID != UNSET)
			{
				HalfEdge leftHENext = GetHalfEdge(leftHE.nextEdgeID, leftThreadInfo);
				HalfEdge leftHENextNext = GetHalfEdge(leftHENext.nextEdgeID, leftThreadInfo);
		
				testVecA = GetVertex(leftHE.startVertexID, leftThreadInfo);
				testVecB = GetVertex(leftHENextNext.startVertexID, leftThreadInfo);
		
				leftHE_Vec = testVecA;
				leftHE_VecID = leftHE.startVertexID;
		
				// left side process
				[allow_uav_condition] [loop]
				while (SideTest(leftBaseVec, rightBaseVec, testVecB) 
				    && InCircleTest(rightBaseVec, leftBaseVec, testVecA, testVecB)// if the next test face is not extist break the loop
				    && leftHENext.twinEdgeID != UNSET)
				{
					// push the twin of the the leftHENextNext
					[branch]
					if (leftHENextNext.twinEdgeID != UNSET) // TODO: fix this problem
						StackPush(leftHENextNext.twinEdgeID,  mergeVInfo.LeftHEStack);
					else
					{
						HalfEdge dummyHE;
						uint dummyHEID = InitNewHalfEdge(leftHE.startVertexID, dummyHE,  leftThreadInfo);
						dummyHE.twinEdgeID = leftHENext.nextEdgeID;
						SetHalfEdge(dummyHE, dummyHEID, leftThreadInfo);
		
						StackPush(dummyHEID,  mergeVInfo.LeftHEStack);
					}
		
					// remove the triangle of the corresponded HE 
					Face face = GetFace(leftHE.faceID);
					face.halfEdgeID =  UNSET;
					SetFace(face,leftHE.faceID);
		

					// move to the next test face
					leftHE_ID = leftHENext.twinEdgeID;
					leftHE = GetHalfEdge(leftHE_ID, leftThreadInfo);
		
					// load and the HE of the next face
					leftHENext = GetHalfEdge(leftHE.nextEdgeID, leftThreadInfo);
					leftHENextNext = GetHalfEdge(leftHENext.nextEdgeID, leftThreadInfo);
		
					// load the new test vecAB
					testVecA = GetVertex(leftHE.startVertexID, leftThreadInfo);
					testVecB = GetVertex(leftHENextNext.startVertexID, leftThreadInfo);
		
					leftHE_Vec = testVecA;
					leftHE_VecID = leftHE.startVertexID;
				}
			}
			else
			{
				HalfEdge leftTwin = GetHalfEdge(leftHE.twinEdgeID, leftThreadInfo);
				leftHE_Vec = GetVertex(leftTwin.startVertexID, leftThreadInfo);
				leftHE_VecID = leftTwin.startVertexID;
			}
		}
		///-- #endregion

		
		
		
		
		
		
		
		
		
		
		
		// right side processing
		///++ #region right side processing
		{
			[branch]
			if (rightHE.nextEdgeID != UNSET)
			{
				HalfEdge rightHENext = GetHalfEdge(rightHE.nextEdgeID, rightThreadInfo);
				HalfEdge rightHENextNext = GetHalfEdge(rightHENext.nextEdgeID, rightThreadInfo);
		
				testVecA = GetVertex(rightHENext.startVertexID, rightThreadInfo);
				testVecB = GetVertex(rightHENextNext.startVertexID, rightThreadInfo);
		
				rightHE_Vec = testVecA;
				rightHE_VecID = rightHENext.startVertexID;
		
				// left side process
				[allow_uav_condition] [loop]
				while (SideTest(leftBaseVec, rightBaseVec, testVecB) 
				    && InCircleTest(rightBaseVec, leftBaseVec, testVecA, testVecB)
				    && rightHENextNext.twinEdgeID != UNSET)
				{
					// push the twin of the the rightHENext
					[branch]
					if (rightHENext.twinEdgeID != UNSET) // TODO: fix this problem
						StackPush(rightHENext.twinEdgeID,  mergeVInfo.RightHEStack);
					else
					{
						HalfEdge dummyHE;
						uint dummyHEID = InitNewHalfEdge(rightHENextNext.startVertexID, dummyHE,  rightThreadInfo);
						dummyHE.twinEdgeID = rightHE.nextEdgeID;
						SetHalfEdge(dummyHE, dummyHEID, rightThreadInfo);
		
						StackPush(dummyHEID,  mergeVInfo.RightHEStack);
					}
		
					// remove the triangle of the corresponded HE 
					Face face = GetFace(rightHE.faceID);
					face.halfEdgeID =  UNSET;
					SetFace(face,rightHE.faceID);
					
					// move to the next test face
					rightHE_ID = rightHENextNext.twinEdgeID;
					rightHE = GetHalfEdge(rightHE_ID, rightThreadInfo);
		
					// load and the HE of the next face
					rightHENext = GetHalfEdge(rightHE.nextEdgeID, rightThreadInfo);
					rightHENextNext = GetHalfEdge(rightHENext.nextEdgeID, rightThreadInfo);
		
					// load the new test vecAB
					testVecA = GetVertex(rightHENext.startVertexID, rightThreadInfo);
					testVecB = GetVertex(rightHENextNext.startVertexID, rightThreadInfo);
		
					rightHE_Vec = testVecA;
					rightHE_VecID = rightHENext.startVertexID;
				}
			}
			else
			{
				HalfEdge rightTwin = GetHalfEdge(rightHE.twinEdgeID, rightThreadInfo);
				rightHE_Vec = GetVertex(rightTwin.startVertexID, rightThreadInfo);
				rightHE_VecID = rightTwin.startVertexID;
			}
		}
		///-- #endregion

		
		
		
		
		
		
		// choice the best side connection
		///++ #region third vector selection
		
		bool rightSideOk = SideTest(leftBaseVec, rightBaseVec, rightHE_Vec);
		bool leftSideOk = SideTest(leftBaseVec, rightBaseVec, leftHE_Vec);
		
		bool forceSide = !(rightSideOk && leftSideOk);
		
		[branch]
		if (rightFinish || leftFinish)
		{
			[branch]
			if (rightFinish)
			{
				thirdVecId = leftHE_VecID;
				[branch]
				if (leftSideOk)
					right = false;
				else
					break;
			}
			else
			{
				thirdVecId = rightHE_VecID;
				[branch]
				if (rightSideOk)
					right = true;
				else
					break;
			}
		}else if (forceSide)
		{
			[branch]
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
			[branch]
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
			CreateTriangle(rightBaseVecId, leftBaseVecId, thirdVecId,
						   he1, he1ID,			    // get the first he
						   he2, he2ID,			    // get the sec he
						   he3, he3ID, 			    // get the third he
						   false, leftThreadInfo, ccw);	// don't store the results
		
		
			// link the new face
			[branch]
			if (right)
			{
		
				// link the he3
				[branch]
				if (rightHE.nextEdgeID != UNSET)
				{
					he3.twinEdgeID = rightHE_ID;
					rightHE.twinEdgeID = he3ID;
				}
				else 
				{
					he3ID = rightHE.twinEdgeID;
				
					// ccw test
					[branch]
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
					prevHE = GetHalfEdge(prevHE_ID,leftThreadInfo);
					
					
					prevHE.twinEdgeID = he1ID;
					he1.twinEdgeID = prevHE_ID;
		
					// save the new prev value
					SetHalfEdge(prevHE, prevHE_ID, leftThreadInfo);
				}
		
				// set the new prev HE
				prevHE_ID = he2ID;
				prevHE = he2;
		
			}
			else
			{
				// link the he2 
				[branch]
				if (leftHE.nextEdgeID != UNSET)
				{
					he2.twinEdgeID = leftHE_ID;
					leftHE.twinEdgeID = he2ID;
				}
				else
				{
					he2ID = leftHE.twinEdgeID;
				
					// ccw test
					[branch]
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
				[branch]
				if (prevHE_ID != UNSET)
				{
					prevHE = GetHalfEdge(prevHE_ID,leftThreadInfo);
					
					prevHE.twinEdgeID = he1ID;
					he1.twinEdgeID = prevHE_ID;
		
					// save the new prev value
					SetHalfEdge(prevHE, prevHE_ID, leftThreadInfo);
				}
		
				// set the new prev HE
				prevHE_ID = he3ID;
				prevHE = he3;
				

			}
			
			// the first time store the first merging
			[branch]
			if(first){
				// set the last HE_ID 
				leftThreadInfo.LeftFirstMergingHEID = he1ID;
				rightThreadInfo.RightFirstMergingHEID = he1ID;
				first = false;
			}		
			

			// create a new face
			uint2 newFaceID =  CreateFace(he1, he1ID,
										  he2, he2ID,
										  he3, he3ID,
										  leftThreadInfo);
		
		
			// store the results
			SetHalfEdge(he1, he1ID, leftThreadInfo);
			SetHalfEdge(he2, he2ID, leftThreadInfo);
			SetHalfEdge(he3, he3ID, leftThreadInfo);
		
			SetHalfEdge(rightHE, rightHE_ID, rightThreadInfo);
			SetHalfEdge(leftHE, leftHE_ID, leftThreadInfo);
		
		}
		
		///-- #endregion

		
		
		
		// pull the next he
		[branch]
		if (right && !rightFinish)
		{
			uint tmpSave = rightHE_ID;
			[branch]
			if (!StackPull(rightHE_ID, mergeVInfo.RightHEStack))
			{
				rightHE_ID = tmpSave;
				rightFinish = true;
			}
		}
		
		[branch]
		if (!right && !leftFinish)
		{
			uint tmpSave = leftHE_ID;
			[branch]
			if (!StackPull(leftHE_ID, mergeVInfo.LeftHEStack))
			{
				leftHE_ID = tmpSave;
				leftFinish = true;
			}
		}
			
		[branch]
		if (rightFinish && leftFinish)
			break;

	}
	
	// set the last HE_ID 
	leftThreadInfo.LeftLastMergingHEID = prevHE_ID;
	rightThreadInfo.RightLastMergingHEID = prevHE_ID;

			
	// ============================================================== Finalize
	// save the merge info
	SetThreadInfo(leftThreadInfo);
	SetThreadInfo(rightThreadInfo);
}


//////////////////////////////////////////////////////////////////////////////////
