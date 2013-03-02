
#include "includes.h"

#ifndef H_IOBuffer_UTILS
#define H_IOBuffer_UTILS


// ------------ Face  ------------ //

static __device__
inline Face GetFace(Face *FaceList, uint index, ThreadInfo *threadInfo)
{
	return FaceList[index + threadInfo->offsetFaceList];
}

static __device__
inline Face GetFace(Face *FaceList, uint2 index, ThreadInfo *threadInfo)
{
	return FaceList[index.x + threadInfo->offsetFaceList];
}

static __device__
inline Face GetFace(Face *FaceList, uint2 index)
{
	return FaceList[index.x + index.y];
}

static __device__
inline void SetFace(Face *FaceList, Face value , uint2 index, ThreadInfo *threadInfo)
{
	FaceList[index.x + threadInfo->offsetFaceList] = value;
}

static __device__
inline void SetFace(Face *FaceList, Face value , uint2 index)
{
	FaceList[index.x + index.y] = value;
}

// ------------ Boundary  ------------ //

static __device__
inline BoundaryNode GetBoundaryNode(BoundaryNode *BoundaryList, uint index, ThreadInfo *threadInfo)
{
	return BoundaryList[index + threadInfo->offsetBoundaryList];
}

static __device__
inline void SetBoundaryNode(BoundaryNode *BoundaryList, BoundaryNode *value , uint index, ThreadInfo *threadInfo)
{
	BoundaryList[index + threadInfo->offsetBoundaryList] = *value;
}

// ------------ Delaunay Stack  ------------ //

static __device__
inline void PushDelaunayNode(DelaunayNode *stack, DelaunayNode newNode , ThreadInfo *threadInfo)
{
	// get the id of the new node
	uint newNodeId = threadInfo->endDNOfStack;
	
	// inc the stack head
	threadInfo->endDNOfStack++;
	
	if(threadInfo->numDNinStack < threadInfo->endDNOfStack){
		threadInfo->numDNinStack=threadInfo->endDNOfStack;
	}
	
	// store the node
	stack[threadInfo->offsetDNStack + newNodeId] = newNode;
}

static __device__
inline bool PullDelaunayNode(DelaunayNode *stack, DelaunayNode *node , ThreadInfo *threadInfo)
{
	bool result = false;
	
	// set to zero
    memset(node,0, sizeof(DelaunayNode));
	
	if(threadInfo->endDNOfStack > 0){
		
		// dec the end offset
		threadInfo->endDNOfStack--;
		
		// dec the num of stack nodes
		//threadInfo.numDNinStack--;
		
		// get the node
		*node = stack[threadInfo->offsetDNStack + threadInfo->endDNOfStack];
		
		// set to true
		result = true;
	}
	
	return result;
}

static __device__
inline void ResetDelaunayStack(ThreadInfo *threadInfo)
{
	threadInfo->endDNOfStack 		=  0; // no DN in stack yet
	//threadInfo.startDNOfStack 	=  0; // no DN in stack yet
	threadInfo->numDNinStack	    =  0; // no DN in stack yet
}


// ------------ generic Stack  ------------ // 

static __device__
inline void StackReset(Stack *stack)
{
	stack->end = 0; // no nodes in stack yet
	stack->start = 0; // no nodes in stack yet
}

static __device__
inline void StackPush(uint *UintStack, uint id, Stack *stack)
{
	if (id == UNSET)
		return;

	// get the id of the new node
	uint newNodeId = stack->end;

	// inc the stack head
	stack->end++;

	// store the node
	UintStack[stack->offset + newNodeId] = id;
}

static __device__
inline bool StackPull(uint *UintStack, uint *id, Stack *stack)
{
	*id = 0;
	
	if (stack->end > 0)
	{
		// dec the stack head
		stack->end--;

		// get the node
		*id = UintStack[stack->offset + stack->end];

		return true;
	}
	
	return false;
}


#endif /* H_IOBuffer_UTILS */









