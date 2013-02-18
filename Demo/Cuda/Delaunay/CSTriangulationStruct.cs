using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using FxMaths.Vector;
using GraphicsEngine.Core;
using System.Runtime.InteropServices;

namespace Delaunay
{
	[StructLayout( LayoutKind.Explicit, Size = 16 )] //16
	public struct cbThreadParam
	{
		[FieldOffset( 0 )]
		// max faces per thread
		public uint maxFacesPerThread;

		[FieldOffset( 4 )]
		// max Half edge per thread
		public uint maxHalfEdgePerThread;

		[FieldOffset( 8 )]
		// max vertex per thread
		public uint maxBoundaryNodesPerThread;

		[FieldOffset( 12 )]
		// num of regions
		public uint RegionsNum;
	}

	
	[StructLayout( LayoutKind.Explicit, Size = 32 )] //20
	public struct csMergeVThreadParam
	{
		[FieldOffset( 0 )]
		// Num of the thread
		public uint ThreadNum;

		[FieldOffset(4)]
		// Num of max elements in stack
		public uint stackMaxSize;

		[FieldOffset(8)]
		// The current depth
		public uint depth;

		[FieldOffset(12)]
		// Num of thread per row
		public uint ThreadNumPerRow;

		[FieldOffset(16)]
		// Num of thread per row
		public uint HorizontalThreadNum;
	}

	[StructLayout(LayoutKind.Explicit, Size = 16)] //4
	public struct csMergeHThreadParam
	{
		[FieldOffset(0)]
		// Num of the thread
		public uint ThreadNum;

		[FieldOffset(4)]
		// The current depth
		public uint depth;

		[FieldOffset(8)]
		// Num of max elements in stack
		public uint stackMaxSize;

		[FieldOffset(12)]
		// Num of thread per row
		public uint ThreadNumPerRow;
	}
	
	
	// ------------ HalfEdge  ------------ //

	public struct csHalfEdge
	{
		//	Start position index
		public uint startVertexID;

		//	twin edge in the triangle
		public uint twinEdgeID;

		//  next edge in the triangle
		public uint nextEdgeID;

		//  face that belong this edge
		public FxVector2i faceID;


		/// <summary>
		/// Get the size of the struct 
		/// </summary>
		/// <returns></returns>
		public static int GetStructSize()
		{
			return 3 * ComputeShader.SizeOfInt1 + ComputeShader.SizeOfInt2;
		}
	};

	// ------------ BoundaryNodes  ------------ //

	public struct csBoundaryNode
	{
		//	The next node on the boundary
		public uint PrevNodeID;

		//	The prev node on the boundary
		public uint NextNodeID;

		//  The edhe in the boundary
		public uint halfEdgeID;

		/// <summary>
		/// Get the size of the struct 
		/// </summary>
		/// <returns></returns>
		public static int GetStructSize()
		{
			return 3 * ComputeShader.SizeOfInt1;
		}
	};

	// ------------ Circle  ------------ //

	public struct csCircle
	{

		// Center
		public FxVector2f center;

		// radius 
		public float radius;

		/// <summary>
		/// Get the size of the struct 
		/// </summary>
		/// <returns></returns>
		public static int GetStructSize()
		{
			return ComputeShader.SizeOfFloat1 + ComputeShader.SizeOfFloat2;
		}

	};

	// ------------ Face  ------------ //

	public struct csFace
	{
		//  Start of the Face
		public uint halfEdgeID;

#if false

#if USE_CIRCLE_TEST

	//  Boundary circle
	Circle boundary;

#else

		// The Min point of the boundary
		public FxVector2f Min;

		// The Max point of the boundary
		public FxVector2f Max;

#endif

#endif

		/// <summary>
		/// Get the size of the struct 
		/// </summary>
		/// <returns></returns>
		public static int GetStructSize()
		{
#if false

#if USE_CIRCLE_TEST
			return ComputeShader.SizeOfInt1 + csCircle.GetStructSize();
#else
			return ComputeShader.SizeOfInt1 + 2 * ComputeShader.SizeOfFloat2;
#endif

#else
            return ComputeShader.SizeOfInt1;
#endif
		}
	}

	// ------------ Region Info  ------------ //

	public struct RegionInfo
	{

        // the offset of the vertex
        public uint VertexOffset;

		// the number of the vertex
		public uint VertexNum;

		/// <summary>
		/// Get the size of the struct 
		/// </summary>
		/// <returns></returns>
		public static int GetStructSize()
		{
			return ComputeShader.SizeOfInt2;
		}
	};

	// ------------ Region Info  ------------ //

	public struct csThreadInfo
	{

		// The id of the thread
		public uint threadID;

		// offset for the face list
		public uint offsetFaceList;

		// offset for the Half edge list
		public uint offsetHalfEdgeList;

		// offset for the Vertex list
		public uint offsetVertexList;

		// offset for the Boundary list
		public uint offsetBoundaryList;

		// offset for the delaunay node 
		public uint offsetDNStack;

		// The Next Face ID (max face id)
		public uint lastFaceID;

		// The Next HalfEdge ID (max HalfEdge id)
		public uint lastHalfEdgeID;

		// The Next Boundary Node ID (max Boundary Node id)
		public uint lastBoundaryNodeID;

		// The rood index of the boundary
		public uint boundaryNodeRootID;

		// the start of the DelaunayNode in the stack
		public uint startDNOfStack;

		// the end of the DelaunayNode in the stack
		public uint endDNOfStack;

		// the num of node in the stack
		public uint numDNinStack;


		// Boundary Min/Max {

			/// The Max boundary node in the X axis
			public uint Boundary_X_MaxID;

			/// The Max boundary node in the Y axis
			public uint Boundary_Y_MaxID;

			/// The Min boundary node in the X axis
			public uint Boundary_X_MinID;

			/// The Min boundary node in the Y axis
			public uint Boundary_Y_MinID;

		//}

			// emergency HE id
			public uint LeftLastMergingHEID;
			public uint RightLastMergingHEID;
			public uint LeftFirstMergingHEID;
			public uint RightFirstMergingHEID;

		/// <summary>
		/// Get the size of the struct 
		/// </summary>
		/// <returns></returns>
		public static int GetStructSize()
		{
			return 20 * ComputeShader.SizeOfInt1 + ComputeShader.SizeOfInt2;
		}
	};

	// ------------ DelaunayNode  ------------ //

	public struct csDelaunayNode
	{

		// the start half edge
		public uint RootHalfEdgeID;

		// the half edge of the next face
		public uint NextFaceHalfEdgeID;

		/// <summary>
		/// Get the size of the struct 
		/// </summary>
		/// <returns></returns>
		public static int GetStructSize()
		{
			return 2 * ComputeShader.SizeOfInt1;
		}
	};


	// ------------ Stack struct  ------------ //

	public struct csStack
	{
		public uint start;
		public uint end;
		public uint offset;

		/// <summary>
		/// Get the size of the struct 
		/// </summary>
		/// <returns></returns>
		public static int GetStructSize()
		{
			return 3 * ComputeShader.SizeOfInt1;
		}
	};

	// ------------ Merge V Info  ------------ //

	public struct csMergeVInfo
	{

		// current thread id
		public uint threadID;

		// stack for the he merging
		public csStack LeftHEStack;
		public csStack RightHEStack;

		/// <summary>
		/// Get the size of the struct 
		/// </summary>
		/// <returns></returns>
		public static int GetStructSize()
		{
			return 1 * ComputeShader.SizeOfInt1 + 2*csStack.GetStructSize();
		}

	};

	// ------------ Merge H Info  ------------ //

	public struct csMergeHInfo
	{

		// current thread id
		public uint threadID;

		// stack for the he merging
		public csStack UpHEStack;
		public csStack DownHEStack;

		// the first left Thread Info
		public uint UpFirstThreadID;
		public uint DownFirstThreadID;
		public uint UpLastThreadID;
		public uint DownLastThreadID;
		
		/// <summary>
		/// Get the size of the struct 
		/// </summary>
		/// <returns></returns>
		public static int GetStructSize()
		{
			return 5 * ComputeShader.SizeOfInt1 + 2 * csStack.GetStructSize();
		}

	};



	// ------------ Merge V Params  ------------ //

	public struct csMergeVParams
	{

		public FxVector2i RegionSidesID;

		/// <summary>
		/// Get the size of the struct 
		/// </summary>
		/// <returns></returns>
		public static int GetStructSize()
		{
			return ComputeShader.SizeOfInt2;
		}

	};




	// ------------ Merge H Params  ------------ //

	public struct csMergeHParams
	{
		/// <summary>
		/// The Start/Stop of the Up region
		/// </summary>
		public FxVector2i UpBN;

		/// <summary>
		/// The start/stop of the Down region
		/// </summary>
		public FxVector2i DownBN;

		/// <summary>
		/// Get the size of the struct 
		/// </summary>
		/// <returns></returns>
		public static int GetStructSize()
		{
			return 2*ComputeShader.SizeOfInt2;
		}

	};
}
