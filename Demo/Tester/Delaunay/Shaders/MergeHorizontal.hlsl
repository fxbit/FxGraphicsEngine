
#include "mh_includes.hlslh"

//////////////////////////////////////////////////////////////////////////////////
// Main 
//////////////////////////////////////////////////////////////////////////////////


[numthreads(1, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID,
		  uint GI : SV_GroupIndex )
{

	uint threadID = (2*threadId.x + 1)*pow(2,depth) - 1;

	if(threadID >= ThreadNum)
		return;
	
	MergeHInfo HInfo;// = GetMergeHInfo(threadID);

	HInfo.UpFirstThreadID = threadID * (ThreadNumPerRow + 1);
	HInfo.DownFirstThreadID = (threadID + 1) * (ThreadNumPerRow + 1);
	HInfo.UpLastThreadID = (threadID + 1) * (ThreadNumPerRow + 1) -1;
	HInfo.DownLastThreadID = (threadID + 2) * (ThreadNumPerRow + 1) - 1;
	
	// init stack
	HInfo.threadID = threadID;
#ifdef USE_SHARED_MEM
	HInfo.UpHEStack.offset = (uint)(2 * GI * stackMaxSize);
	HInfo.DownHEStack.offset = (uint)((2 * GI + 1) * stackMaxSize);
#else
	HInfo.UpHEStack.offset = (uint)(2 * threadID * stackMaxSize);
	HInfo.DownHEStack.offset = (uint)((2 * threadID + 1) * stackMaxSize);
#endif
	
	// get all the threadd info
	ThreadInfo UpFirstThreadInfo = GetThreadInfo(HInfo.UpFirstThreadID);
	ThreadInfo DownFirstThreadInfo = GetThreadInfo(HInfo.DownFirstThreadID);
	ThreadInfo UpLastThreadInfo = GetThreadInfo(HInfo.UpLastThreadID);
	ThreadInfo DownLastThreadInfo = GetThreadInfo(HInfo.DownLastThreadID);

	float UpVertexRange = UpLastThreadInfo.offsetVertexList - UpFirstThreadInfo.offsetVertexList;
	float DownVertexRange = DownLastThreadInfo.offsetVertexList - DownFirstThreadInfo.offsetVertexList;
	
	float UpThreadRange =HInfo.UpLastThreadID - HInfo.UpFirstThreadID;
	float DownThreadRange = HInfo.DownLastThreadID - HInfo.DownFirstThreadID;


	// get the start/end HE
	uint UpStartHEID = GetBoundaryNode(GetBoundaryNode(UpFirstThreadInfo.Boundary_X_MinID, UpFirstThreadInfo).PrevNodeID, UpFirstThreadInfo).halfEdgeID;
	HalfEdge UpStartHE = GetHalfEdge(UpStartHEID,UpFirstThreadInfo);
	
	// check that the HE is in boundary
	[allow_uav_condition] [loop]
	while (UpStartHE.twinEdgeID != UNSET)
	{
		HalfEdge tmpUpStartHETwin = GetHalfEdge(UpStartHE.twinEdgeID,UpFirstThreadInfo);
		HalfEdge tmpUpStartHETwinNext = GetHalfEdge(tmpUpStartHETwin.nextEdgeID,UpFirstThreadInfo);
		UpStartHEID = tmpUpStartHETwinNext.nextEdgeID;
		UpStartHE = GetHalfEdge(UpStartHEID,UpFirstThreadInfo);
	}
	
	uint UpEndHEID = GetBoundaryNode(UpLastThreadInfo.Boundary_X_MaxID, UpLastThreadInfo).halfEdgeID;
	HalfEdge UpEndHE = GetHalfEdge(UpEndHEID,UpLastThreadInfo);
	
	// check that the HE is in boundary
	[allow_uav_condition] [loop]
	while (UpEndHE.twinEdgeID != UNSET)
	{
		HalfEdge tmpUpEndHETwin = GetHalfEdge(UpEndHE.twinEdgeID,UpLastThreadInfo);		
		UpEndHEID = tmpUpEndHETwin.nextEdgeID;
		UpEndHE = GetHalfEdge(UpEndHEID,UpLastThreadInfo);
	}

	
	uint DownStartHEID = GetBoundaryNode(DownFirstThreadInfo.Boundary_X_MinID, DownFirstThreadInfo).halfEdgeID;
	HalfEdge DownStartHE = GetHalfEdge(DownStartHEID,DownFirstThreadInfo);
	
	// check that the HE is in boundary
	[allow_uav_condition] [loop]
	while (DownStartHE.twinEdgeID != UNSET)
	{
		HalfEdge tmpDownStartHETwin = GetHalfEdge(DownStartHE.twinEdgeID, DownFirstThreadInfo);
		DownStartHEID = tmpDownStartHETwin.nextEdgeID;
		DownStartHE = GetHalfEdge(DownStartHEID, DownFirstThreadInfo);
	}
	
	uint DownEndHEID = GetBoundaryNode(GetBoundaryNode(DownLastThreadInfo.Boundary_X_MaxID, DownLastThreadInfo).PrevNodeID, DownLastThreadInfo).halfEdgeID;
	HalfEdge DownEndHE = GetHalfEdge(DownEndHEID, DownLastThreadInfo);

	// check that the HE is in boundary
	[allow_uav_condition] [loop]
	while (DownEndHE.twinEdgeID != UNSET)
	{
		HalfEdge tmpDownEndHETwin = GetHalfEdge(DownEndHE.twinEdgeID,DownLastThreadInfo);
		HalfEdge tmpDownEndHETwinNext = GetHalfEdge(tmpDownEndHETwin.nextEdgeID,DownLastThreadInfo);
		DownEndHEID = tmpDownEndHETwinNext.nextEdgeID;
		DownEndHE = GetHalfEdge(DownEndHEID,DownLastThreadInfo);
	}

	{
		// load the Correspond face
		Face tmpFace = GetFace(DownEndHE.faceID);
	
		// check if the face have be deleted
		[branch]
		if (tmpFace.halfEdgeID == UNSET)
		{
			DownEndHEID = DownLastThreadInfo.RightLastMergingHEID;
			DownEndHE = GetHalfEdge(DownEndHEID, DownLastThreadInfo);
		}
	}
	
	{
		// load the Correspond face
		Face tmpFace = GetFace(DownStartHE.faceID);
	
		// check if the face have be deleted
		[branch]
		if (tmpFace.halfEdgeID == UNSET)
		{
			DownStartHEID = DownFirstThreadInfo.LeftLastMergingHEID;
			DownStartHE = GetHalfEdge(DownStartHEID, DownFirstThreadInfo);
		}
	}
	
	{
		// load the Correspond face
		Face tmpFace = GetFace(UpEndHE.faceID);
	
		// check if the face have be deleted
		[branch]
		if (tmpFace.halfEdgeID == UNSET)
		{
			UpEndHEID = UpLastThreadInfo.RightFirstMergingHEID;
			UpEndHE = GetHalfEdge(UpEndHEID, UpLastThreadInfo);
		}
	}
	
	{
		// load the Correspond face
		Face tmpFace = GetFace(UpStartHE.faceID);
	
		// check if the face have be deleted
		if (tmpFace.halfEdgeID == UNSET)
		{
			UpStartHEID = UpFirstThreadInfo.LeftFirstMergingHEID;
			UpStartHE = GetHalfEdge(UpStartHEID, UpFirstThreadInfo);
		}
	}
	
	// clean the stacks for the merging
	StackReset(HInfo.UpHEStack);
	StackReset(HInfo.DownHEStack);


// ============================================================== Tracking

	int SafeCount = stackMaxSize;
	
	
	///++ #region Find all the up path


	// load the tmp 
	uint tmpUpHEID = UpStartHEID;
	HalfEdge tmpUpHE = UpStartHE;

	// pass all the up path
	[allow_uav_condition] [loop]
	while (tmpUpHEID != UpEndHEID)
	{
		// push the he
		StackPush(tmpUpHEID, HInfo.UpHEStack);

		uint tmpTmpHEID = tmpUpHEID;
		uint correctID = tmpTmpHEID;
		HalfEdge tmpTmpHE = GetHalfEdge(tmpTmpHEID, UpFirstThreadInfo);
		HalfEdge tmpTmpHENext = GetHalfEdge(tmpTmpHE.nextEdgeID, UpFirstThreadInfo);
		HalfEdge tmpTmpHENextNext = GetHalfEdge(tmpTmpHENext.nextEdgeID, UpFirstThreadInfo);

		

		// check if we are in a corner
		[branch]
		if (tmpTmpHENextNext.twinEdgeID == UNSET)
		{
			// push the cornel and continue from there
			correctID = tmpTmpHENext.nextEdgeID;
		}
		else
		{
			tmpTmpHEID = tmpTmpHENextNext.twinEdgeID;
			
			// move to the next boundary he
			[allow_uav_condition] [loop]
			while (tmpTmpHENextNext.twinEdgeID != UNSET && tmpTmpHENextNext.nextEdgeID != UNSET)
			{

				// try to find the next HE
				tmpTmpHE = GetHalfEdge(tmpTmpHEID, UpFirstThreadInfo);
				tmpTmpHENext = GetHalfEdge(tmpTmpHE.nextEdgeID, UpFirstThreadInfo);
				tmpTmpHENextNext = GetHalfEdge(tmpTmpHENext.nextEdgeID, UpFirstThreadInfo);

				correctID = tmpTmpHENext.nextEdgeID;
				tmpTmpHEID = tmpTmpHENextNext.twinEdgeID;
			}
		}
		
		// set the new tmp HE
		tmpUpHEID = correctID;
	}

	// push the end he
	StackPush(tmpUpHEID, HInfo.UpHEStack);



	///-- #endregion



	///++ #region Find all the down path


	// load the tmp 
	uint tmpDownHEID = DownStartHEID;
	HalfEdge tmpDownHE = DownStartHE;
	SafeCount = stackMaxSize;

	// pass all the down path
	[allow_uav_condition] [loop]
	while (tmpDownHEID != DownEndHEID)
	{
		// push the he
		StackPush(tmpDownHEID, HInfo.DownHEStack);

		uint tmpTmpHEID = tmpDownHEID;
		uint correctID = tmpTmpHEID;
		HalfEdge tmpTmpHE = GetHalfEdge(tmpTmpHEID, UpFirstThreadInfo);
		HalfEdge tmpTmpHENext = GetHalfEdge(tmpTmpHE.nextEdgeID, UpFirstThreadInfo);

		// check if we are in a corner
		[branch]
		if (tmpTmpHENext.twinEdgeID == UNSET)
		{
			// push the cornel and continue from there
			correctID = tmpTmpHE.nextEdgeID;
		}
		else
		{
			tmpTmpHEID = tmpTmpHENext.twinEdgeID;

			// move to the next boundary he
			[allow_uav_condition] [loop]
			while (tmpTmpHENext.twinEdgeID != UNSET && tmpTmpHENext.nextEdgeID != UNSET)
			{

				// try to find the next HE
				tmpTmpHE = GetHalfEdge(tmpTmpHEID, UpFirstThreadInfo);
				tmpTmpHENext = GetHalfEdge(tmpTmpHE.nextEdgeID, UpFirstThreadInfo);

				correctID = tmpTmpHE.nextEdgeID;
				tmpTmpHEID = tmpTmpHENext.twinEdgeID;
			}
		}
		
		// set the new tmp HE
		tmpDownHEID = correctID;
	}

	// push the end he
	StackPush(tmpDownHEID, HInfo.DownHEStack);



	///-- #endregion

	
	
	// ============================================================== Fix the Starting Points
	if(false)
	{
		// Push the base HE
		uint local_UpHE_ID, local_DownHE_ID;
		StackPull(local_UpHE_ID, HInfo.UpHEStack);
		StackPull(local_DownHE_ID, HInfo.DownHEStack);
	
		// get the edge of each side
		HalfEdge local_UpHE = GetHalfEdge(local_UpHE_ID, UpFirstThreadInfo);
		HalfEdge local_DownHE = GetHalfEdge(local_DownHE_ID, DownFirstThreadInfo);
	
		HalfEdge UpHENextHE = GetHalfEdge(local_UpHE.nextEdgeID, UpFirstThreadInfo);
		HalfEdge DownHENextHE = GetHalfEdge(local_DownHE.nextEdgeID, DownFirstThreadInfo);
	
		// get the vertexes
		float2 UpVert = GetVertex(local_UpHE.startVertexID, UpFirstThreadInfo);
		float2 DownVert = GetVertex(local_DownHE.startVertexID, DownFirstThreadInfo);
	
		float2 UpHENextVert = GetVertex(UpHENextHE.startVertexID, UpFirstThreadInfo);
		float2 DownHENextVert = GetVertex(DownHENextHE.startVertexID, DownFirstThreadInfo);
	
		bool UpTest = !SideTest(UpHENextVert, UpVert, DownVert);
		bool DownTest = SideTest(DownVert, DownHENextVert, UpVert);
	
		[allow_uav_condition] [loop]
		while (!(UpTest && DownTest))
		{
			[branch]
			if (!UpTest)
			{
				[branch]
				if (StackPull(local_UpHE_ID, HInfo.UpHEStack))
				{
					local_UpHE = GetHalfEdge(local_UpHE_ID, UpFirstThreadInfo);
					UpHENextHE = GetHalfEdge(local_UpHE.nextEdgeID, UpFirstThreadInfo);
					UpVert = GetVertex(local_UpHE.startVertexID, UpFirstThreadInfo);
					UpHENextVert = GetVertex(UpHENextHE.startVertexID, UpFirstThreadInfo);
				}
				else
					break;
			}
	
			[branch]
			if (!DownTest)
			{
				[branch]
				if (StackPull(local_DownHE_ID, HInfo.DownHEStack))
				{
					local_DownHE = GetHalfEdge(local_DownHE_ID, DownFirstThreadInfo);
					DownHENextHE = GetHalfEdge(local_DownHE.nextEdgeID, DownFirstThreadInfo);
					DownVert = GetVertex(local_DownHE.startVertexID, DownFirstThreadInfo);
					DownHENextVert = GetVertex(DownHENextHE.startVertexID, DownFirstThreadInfo);
				}
				else
					break;
			}
	
			UpTest = !SideTest(UpHENextVert, UpVert, DownVert);
			DownTest = SideTest(DownVert, DownHENextVert, UpVert);
		}
	
	
		// push the old 
		StackPush(local_DownHE_ID, HInfo.DownHEStack);
		StackPush(local_UpHE_ID, HInfo.UpHEStack);
	}


	// ============================================================== Merging 
	//// start the merging 
	ThreadInfo UpThreadInfo = UpFirstThreadInfo;
	ThreadInfo DownThreadInfo = DownFirstThreadInfo;

	// init the he
	uint DownHE_ID, UpHE_ID;
	StackPull(DownHE_ID, HInfo.DownHEStack);
	StackPull(UpHE_ID, HInfo.UpHEStack);

	HalfEdge DownHE = GetHalfEdge(DownHE_ID, DownThreadInfo);
	HalfEdge UpHE = GetHalfEdge(UpHE_ID, UpThreadInfo);

	// get the vertex of the base line
	uint UpBaseVecId = UpHE.startVertexID;
	float2 UpBaseVec = GetVertex(UpBaseVecId, UpThreadInfo);
	uint DownBaseVecId = GetHalfEdge(DownHE.nextEdgeID, DownThreadInfo).startVertexID;
	float2 DownBaseVec = GetVertex(DownBaseVecId, UpThreadInfo);
	float2 testVecA, testVecB;

	uint prevHE_ID = UNSET;
	HalfEdge prevHE;
	bool UpFinish = false, DownFinish = false;
	
	[allow_uav_condition] [loop]
	while (true)
	{
		float2 DownHE_Vec, UpHE_Vec;
		uint DownHE_VecID, UpHE_VecID;
		uint thirdVecId;
		bool Up;

		DownHE = GetHalfEdge(DownHE_ID, DownThreadInfo);
		UpHE = GetHalfEdge(UpHE_ID, UpThreadInfo);

		///++ #region Select the base vec


		// get the vertex of the base line
		[branch]
		if (UpFinish)
		{
			[branch]
			if (UpHE.nextEdgeID != UNSET)
			{
				HalfEdge UpHENext = GetHalfEdge(UpHE.nextEdgeID, UpThreadInfo);
				UpBaseVec = GetVertex(UpHENext.startVertexID, UpThreadInfo);
				UpBaseVecId = UpHENext.startVertexID;
			}
			else
			{
				HalfEdge UpTwin = GetHalfEdge(UpHE.twinEdgeID, UpThreadInfo);
				UpBaseVec = GetVertex(UpTwin.startVertexID, UpThreadInfo);
				UpBaseVecId = UpTwin.startVertexID;
			}
		}
		else
		{
			[branch]
			if (UpHE.nextEdgeID != UNSET)
			{
				UpBaseVecId = UpHE.startVertexID;
				UpBaseVec = GetVertex(UpBaseVecId, UpThreadInfo);
			}
			else
			{
				UpBaseVec = GetVertex(UpHE.startVertexID, UpThreadInfo);
				UpBaseVecId = UpHE.startVertexID;
			}
		}

		[branch]
		if (DownFinish)
		{
			[branch]
			if (DownHE.nextEdgeID != UNSET)
			{
				DownBaseVec = GetVertex(DownHE.startVertexID, DownThreadInfo);
				DownBaseVecId = DownHE.startVertexID;
			}
			else
			{
				HalfEdge DownTwin = GetHalfEdge(DownHE.twinEdgeID, DownThreadInfo);
				DownBaseVec = GetVertex(DownTwin.startVertexID, DownThreadInfo);
				DownBaseVecId = DownTwin.startVertexID;
			}
		}
		else
		{
			[branch]
			if (DownHE.nextEdgeID != UNSET)
			{
				DownBaseVecId = GetHalfEdge(DownHE.nextEdgeID, DownThreadInfo).startVertexID;
				DownBaseVec = GetVertex(DownBaseVecId, UpThreadInfo);
			}
			else
			{
				HalfEdge DownTwin = GetHalfEdge(DownHE.twinEdgeID, DownThreadInfo);
				DownBaseVec = GetVertex(DownTwin.startVertexID, DownThreadInfo);
				DownBaseVecId = DownTwin.startVertexID;
			}
		}
		///-- #endregion

		// Down side processing
		///++ #region Down side processing
		{
			[branch]
			if (DownHE.nextEdgeID != UNSET)
			{
				HalfEdge DownHENext = GetHalfEdge(DownHE.nextEdgeID, DownThreadInfo);
				HalfEdge DownHENextNext = GetHalfEdge(DownHENext.nextEdgeID, DownThreadInfo);

				testVecA = GetVertex(DownHE.startVertexID, DownThreadInfo);
				testVecB = GetVertex(DownHENextNext.startVertexID, DownThreadInfo);

				DownHE_Vec = testVecA;
				DownHE_VecID = DownHE.startVertexID;

				// Down side process
				[allow_uav_condition] [loop]
				while (SideTest(DownBaseVec, UpBaseVec, testVecB) && InCircleTest(UpBaseVec, DownBaseVec, testVecA, testVecB))
				{

					// push the twin of the the DownHENextNext
					[branch]
					if (DownHENextNext.twinEdgeID != UNSET) // TODO: fix this problem
						StackPush(DownHENextNext.twinEdgeID, HInfo.DownHEStack);
					else
					{
						HalfEdge dummyHE;
						uint dummyHEID = InitNewHalfEdge(DownHE.startVertexID, dummyHE, DownThreadInfo);
						dummyHE.twinEdgeID = DownHENext.nextEdgeID;
						SetHalfEdge(dummyHE, dummyHEID, DownThreadInfo);

						StackPush(dummyHEID, HInfo.DownHEStack);
					}

					// remove the triangle of the corresponded HE 
					[branch]
					if (DownHE.faceID.x != UNSET)
					{
						Face face = GetFace(DownHE.faceID);
						face.halfEdgeID =  UNSET;
						SetFace(face,DownHE.faceID);
					}
					
					[branch]
					if (DownHENext.twinEdgeID != UNSET)
					{
						// move to the next test face
						DownHE_ID = DownHENext.twinEdgeID;
						DownHE = GetHalfEdge(DownHE_ID, DownThreadInfo);

						// load and the HE of the next face
						DownHENext = GetHalfEdge(DownHE.nextEdgeID, DownThreadInfo);
						DownHENextNext = GetHalfEdge(DownHENext.nextEdgeID, DownThreadInfo);

						// load the new test vecAB
						testVecA = GetVertex(DownHE.startVertexID, DownThreadInfo);
						testVecB = GetVertex(DownHENextNext.startVertexID, DownThreadInfo);

						DownHE_Vec = testVecA;
						DownHE_VecID = DownHE.startVertexID;
					}
					else
						break;
				}
			}
			else
			{
				HalfEdge DownTwin = GetHalfEdge(DownHE.twinEdgeID, DownThreadInfo);
				DownHE_Vec = GetVertex(DownTwin.startVertexID, DownThreadInfo);
				DownHE_VecID = DownTwin.startVertexID;
			}
		}
		///-- #endregion

		// Up side processing
		///++ #region Up side processing
		{
			[branch]
			if (UpHE.nextEdgeID != UNSET)
			{
				HalfEdge UpHENext = GetHalfEdge(UpHE.nextEdgeID, UpThreadInfo);
				HalfEdge UpHENextNext = GetHalfEdge(UpHENext.nextEdgeID, UpThreadInfo);

				testVecA = GetVertex(UpHENext.startVertexID, UpThreadInfo);
				testVecB = GetVertex(UpHENextNext.startVertexID, UpThreadInfo);

				UpHE_Vec = testVecA;
				UpHE_VecID = UpHENext.startVertexID;

				// Down side process
				[allow_uav_condition] [loop]
				while (SideTest(DownBaseVec, UpBaseVec, testVecB) && InCircleTest(UpBaseVec, DownBaseVec, testVecA, testVecB))
				{
					// push the twin of the the UpHENext
					[branch]
					if (UpHENext.twinEdgeID != UNSET)
					{
						if (UpHENextNext.twinEdgeID != UNSET)
						{
							StackPush(UpHENext.twinEdgeID, HInfo.UpHEStack);
						}
					}
					else
					{
						HalfEdge dummyHE;
						int dummyHEID = InitNewHalfEdge(UpHENextNext.startVertexID, dummyHE, UpThreadInfo);
						dummyHE.twinEdgeID = UpHE.nextEdgeID;
						SetHalfEdge(dummyHE, dummyHEID, UpThreadInfo);

						StackPush(dummyHEID, HInfo.UpHEStack);
					}

					// remove the triangle of the corresponded HE 
					Face face = GetFace(UpHE.faceID);
					face.halfEdgeID =  UNSET;
					SetFace(face,UpHE.faceID);
					
					[branch]
					if (UpHENextNext.twinEdgeID != UNSET)
					{
						
						// move to the next test face
						UpHE_ID = UpHENextNext.twinEdgeID;
						UpHE = GetHalfEdge(UpHE_ID, UpThreadInfo);

						// load and the HE of the next face
						UpHENext = GetHalfEdge(UpHE.nextEdgeID, UpThreadInfo);
						UpHENextNext = GetHalfEdge(UpHENext.nextEdgeID, UpThreadInfo);

						// load the new test vecAB
						testVecA = GetVertex(UpHENext.startVertexID, UpThreadInfo);
						testVecB = GetVertex(UpHENextNext.startVertexID, UpThreadInfo);

						UpHE_Vec = testVecA;
						UpHE_VecID = UpHENext.startVertexID;
					}
					else
						break;
				}
			}
			else
			{
				HalfEdge UpTwin = GetHalfEdge(UpHE.twinEdgeID, UpThreadInfo);
				UpHE_Vec = GetVertex(UpTwin.startVertexID, UpThreadInfo);
				UpHE_VecID = UpTwin.startVertexID;
			}
		}
		///-- #endregion

		// choice the best side connection
		///++ #region third vector selection

		bool UpSideOk = SideTest(DownBaseVec, UpBaseVec, UpHE_Vec);
		bool DownSideOk = SideTest(DownBaseVec, UpBaseVec, DownHE_Vec);

		bool forceSide = !(UpSideOk && DownSideOk);

		[branch]
		if (UpFinish || DownFinish)
		{
			[branch]
			if (UpFinish)
			{
				thirdVecId = DownHE_VecID;
				[branch]
				if (DownSideOk)
					Up = false;
				else
					break;
			}
			else
			{
				thirdVecId = UpHE_VecID;
				[branch]
				if (UpSideOk)
					Up = true;
				else
					break;
			}
		}
		else if (forceSide)
		{
			[branch]
			if (!UpSideOk)
			{
				thirdVecId = DownHE_VecID;
				Up = false;
			}
			else
			{
				thirdVecId = UpHE_VecID;
				Up = true;
			}
		}
		else
		{
			[branch]
			if (InCircleTest(DownBaseVec, UpBaseVec, DownHE_Vec, UpHE_Vec))
			{
				thirdVecId = UpHE_VecID;
				Up = true;
			}
			else
			{
				thirdVecId = DownHE_VecID;
				Up = false;
			}
		}

		///-- #endregion

		// select the UP/Down threadInfo
		
		int DownOffset =  clamp((int)floor((float)(DownThreadRange)*(float)(DownHE.startVertexID - DownFirstThreadInfo.offsetVertexList) / (float)DownVertexRange) - 1,0,DownThreadRange);
		int UpOffset = clamp((int)floor((float)(UpThreadRange) * (float)(UpHE.startVertexID - UpFirstThreadInfo.offsetVertexList) / (float)UpVertexRange) - 1,0,UpThreadRange);

		UpThreadInfo = GetThreadInfo((uint)(HInfo.UpFirstThreadID + UpOffset));
		DownThreadInfo = GetThreadInfo((uint)(HInfo.DownFirstThreadID + DownOffset));

		///++ #region Create the Face

		{
			// create 3 new he
			HalfEdge he1, he2, he3;
			uint he1ID, he2ID, he3ID;
			bool ccw;
			CreateTriangle(UpBaseVecId, DownBaseVecId, thirdVecId,
						   he1, he1ID,			    // get the first he
						   he2, he2ID,			    // get the sec he
						   he3, he3ID, 			    // get the third he
						   false,  UpThreadInfo, ccw);	// don't store the results


			// link the new face
			[branch]
			if (Up)
			{

				// link the he3
				he3.twinEdgeID = UpHE_ID;
				UpHE.twinEdgeID = he3ID;

				// link the he1 with the prev he
				[branch]
				if (prevHE_ID != UNSET)
				{
					prevHE.twinEdgeID = he1ID;
					he1.twinEdgeID = prevHE_ID;

					// save the new prev value
					SetHalfEdge(prevHE, prevHE_ID, UpThreadInfo);
				}

				// set the new prev HE
				prevHE_ID = he2ID;
				prevHE = he2;

			}
			else
			{
				// link the he2 
				he2.twinEdgeID = DownHE_ID;
				DownHE.twinEdgeID = he2ID;

				// link the he1 with the prev he
				[branch]
				if (prevHE_ID != UNSET)
				{
					prevHE.twinEdgeID = he1ID;
					he1.twinEdgeID = prevHE_ID;

					// save the new prev value
					SetHalfEdge(prevHE, prevHE_ID, UpThreadInfo);
				}

				// set the new prev HE
				prevHE_ID = he3ID;
				prevHE = he3;
			}


			// create a new face
			uint2 newFaceID = CreateFace(he1, he1ID,
									     he2, he2ID,
										 he3, he3ID,
										 UpThreadInfo);


			// store the results
			SetHalfEdge(he1, he1ID, UpThreadInfo);
			SetHalfEdge(he2, he2ID, UpThreadInfo);
			SetHalfEdge(he3, he3ID, UpThreadInfo);

			SetHalfEdge(UpHE, UpHE_ID, UpThreadInfo);
			SetHalfEdge(DownHE, DownHE_ID, UpThreadInfo);
		}

		///-- #endregion

		SetThreadInfo(UpThreadInfo);
		SetThreadInfo(DownThreadInfo);
		
		// pull the next he
		[branch]
		if (Up && !UpFinish)
		{
			uint tmpSave = UpHE_ID;
			[branch]
			if (!StackPull(UpHE_ID, HInfo.UpHEStack))
			{
				UpHE_ID = tmpSave;
				UpFinish = true;
			}
		}

		[branch]
		if (!Up && !DownFinish)
		{
			uint tmpSave = DownHE_ID;
			[branch]
			if (!StackPull(DownHE_ID, HInfo.DownHEStack))
			{
				DownHE_ID = tmpSave;
				DownFinish = true;
			}
		}

		[branch]
		if (UpFinish && DownFinish)
			break;

	}


	// ============================================================== Finishing
	
	
	//SetMergeHInfo(HInfo);
	return;
	
}


//////////////////////////////////////////////////////////////////////////////////
