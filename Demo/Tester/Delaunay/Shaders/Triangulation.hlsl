
#include "tr_includes.hlslh"


void Init(inout ThreadInfo threadInfo)
{
	uint bn1_ID, bn2_ID, bn3_ID;
	BoundaryNode bn1,bn2,bn3;
	HalfEdge he1,he2,he3;
	uint he1ID, he2ID, he3ID;
	uint offset =  threadInfo.offsetVertexList;
	bool ccw;
	
	// get the 3 first points and create a triangle
	// create the first triangle
	CreateTriangle(offset, offset+1 , offset+2 ,			// the 3 first vertex
				   he1, he1ID,			// get the first he
				   he2, he2ID,			// get the sec he
				   he3, he3ID, 			// get the third he
			       true, threadInfo, ccw);	// store the results
			
	// create a new face that start from HalfEdge1ID
	CreateFace(he1ID,  threadInfo);
	
	// create and link the boundary list   
	{
		
		// read again the face that created from the 0,1,2 vertex ... CCW problems
		bn1_ID = InitNewBoundaryNode(he1ID, bn1, threadInfo);			//  init the root bn
		bn2_ID = InitNewBoundaryNode(he1.nextEdgeID, bn2, threadInfo);	//  init the second bn
		he2 = GetHalfEdge(he1.nextEdgeID, threadInfo);					//  move to the next he
		bn3_ID = InitNewBoundaryNode(he2.nextEdgeID, bn3, threadInfo);	//  init the third bn
		he3 = GetHalfEdge(he2.nextEdgeID, threadInfo);					//  move to the next he

		// set the root bn to 1
		threadInfo.boundaryNodeRootID = bn1_ID;

		// link the root with the next node
		bn1.NextNodeID = bn2_ID;
		bn2.PrevNodeID = bn1_ID;

		// link the root with the next node
		bn2.NextNodeID = bn3_ID;
		bn3.PrevNodeID = bn2_ID;

		// link the first with the last one
		bn3.NextNodeID = bn1_ID;
		bn1.PrevNodeID = bn3_ID;


		// store the new bn
		SetBoundaryNode(bn1 , bn1_ID, threadInfo);
		SetBoundaryNode(bn2 , bn2_ID, threadInfo);
		SetBoundaryNode(bn3 , bn3_ID, threadInfo);

		/*
		// init the BN Min/Max
		threadInfo.Boundary_X_MaxID = bn1_ID;
		threadInfo.Boundary_Y_MaxID = bn1_ID;
		threadInfo.Boundary_X_MinID = bn1_ID;
		threadInfo.Boundary_Y_MinID = bn1_ID;

		// update the BN
		float2 vert = GetVertex(he2.startVertexID, threadInfo);
		UpdateBoundaryMinMax(bn2_ID, vert, threadInfo);
		vert = GetVertex(he3.startVertexID, threadInfo);
		UpdateBoundaryMinMax(bn3_ID, vert, threadInfo);
		*/

	}

}


[numthreads(128, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
	// check if we can process
	if(threadId.x >= RegionsNum)
		return;

	int i=0, VertexNum;
	ThreadInfo threadInfo = (ThreadInfo)0;
	
	
	// init the thread infos
	threadInfo.threadID = threadId.x;
	
	RegionInfo regionInfo = GetRegionInfo(threadInfo.threadID);
	
	threadInfo.offsetFaceList		=  threadId.x * maxFacesPerThread;
	threadInfo.offsetHalfEdgeList	=  threadId.x * maxHalfEdgePerThread;
	threadInfo.offsetVertexList		=  regionInfo.VertexOffset;
	threadInfo.offsetBoundaryList	=  threadId.x * maxBoundaryNodesPerThread;
	threadInfo.offsetDNStack		=  threadId.x * MAX_FACE_CORRECTIONS;
	
	threadInfo.lastFaceID			=  float2(0,threadInfo.offsetFaceList); 	// no face yet
	threadInfo.lastHalfEdgeID		=  threadInfo.offsetHalfEdgeList; 	// no he yet
	threadInfo.lastBoundaryNodeID	=  0; 	// no bn yet
	threadInfo.boundaryNodeRootID	=  0; 	// set the node root to unset;
	threadInfo.endDNOfStack 		=  0; 	// no DN in stack yet
	threadInfo.startDNOfStack 		=  0; 	// no DN in stack yet
	threadInfo.numDNinStack			=  0; 	// no DN in stack yet
	
	// init the triangulation by create the first triangle
	Init(threadInfo);

	// read the number of vertex for this thread
	VertexNum = regionInfo.VertexNum + threadInfo.offsetVertexList;

	// start from the 4th vertex and after
	i=3 + threadInfo.offsetVertexList;

	// ==============================================================   Add the Vertex
	// loop all the vertex
	[loop] while(i<VertexNum)
	{
		// get the specific vertex 
		float2 vert = GetVertex(i, threadInfo);

		// reset the stack of the delaunay 
		ResetDelaunayStack(threadInfo);
		
		Face hitFace;
		uint hitFaceID;
		
		// try to find the face that include the face
		/*
		if( FindFace(vert, hitFace, hitFaceID, threadInfo) ){
			uint heMinDistID;

			// get the min dist from the edges of the face
			float minDist = FaceMinDis(hitFace, vert, heMinDistID, threadInfo);

			// check if the vert is very close to he in that case handle different
			if(minDist > 0.1)
			{
				// split the face in 3 different faces
				SplitFace( hitFace, hitFaceID,  // the face that we have hit
						   vert, i, 			// the new vertex
						   threadInfo);

			}else{
				
				// TODO: split face in 4 parts
			}

		}else{ /* !FindFace */
			
			// now increase the boundary
			
			// Add outside vertex (using boundary nodes)
			AddOutsideVertex( vert, i, 			// the new vertex
							  threadInfo);

		//}
		
		// fix all the new faces
		FixStackFaces(threadInfo);
		
		// inc i
		i++;
	}
	
	// ==============================================================   Prepare the Right merging state
	
	// calc the start and end of the right boundary
	//CalcSideRightBoundary(threadInfo);
	
	// calc the start and end of the left boundary
	//CalcSideLeftBoundary(threadInfo);
	
	{
		uint tmpNodeID = threadInfo.boundaryNodeRootID;
		BoundaryNode tmpNode = GetBoundaryNode(tmpNodeID,threadInfo);
		float2 min = GetVertex( GetHalfEdge( tmpNode.halfEdgeID, threadInfo).startVertexID, threadInfo);
		float2 max = min;
		
		uint2 minID = float2(tmpNodeID,tmpNodeID);
		uint2 maxID = minID;
		
		// calc the Y min bn
		[allow_uav_condition]  while(true){
			
			uint heID = tmpNode.halfEdgeID;
			HalfEdge he = GetHalfEdge(heID, threadInfo);
			float2 he_vert = GetVertex(he.startVertexID, threadInfo);
			
			// check for y min
			[branch] if( he_vert.y < min.y ){
				min.y = he_vert.y;
				minID.y = tmpNodeID;
			}
			
			// check for y max
			[branch] if( he_vert.y > max.y ){
				max.y = he_vert.y;
				maxID.y = tmpNodeID;
			}
			
			// check for x min
			[branch] if( he_vert.x < min.x ){
				min.x = he_vert.x;
				minID.x = tmpNodeID;
			}
			
			// check for x max
			[branch] if( he_vert.x > max.x ){
				max.x = he_vert.x;
				maxID.x = tmpNodeID;
			}
			
			// move to the next node
			tmpNodeID = tmpNode.NextNodeID;
			tmpNode = GetBoundaryNode(tmpNodeID, threadInfo);
			
			// check if we are in the start
			[branch] if( tmpNodeID == threadInfo.boundaryNodeRootID)
				break;
		}
		
		threadInfo.Boundary_Y_MinID = minID.y;
		threadInfo.Boundary_Y_MaxID = maxID.y;
		threadInfo.Boundary_X_MinID = minID.x;
		threadInfo.Boundary_X_MaxID = maxID.x;
	}
	
	// ==============================================================   Store the thread info results
	
	// store the thread info here for debug.
	SetThreadInfo(threadInfo);
}
