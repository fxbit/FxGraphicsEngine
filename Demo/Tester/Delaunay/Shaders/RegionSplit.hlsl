


#define THREAD_NUM 128
#define THREAD_NUM_V_Y 8
#define DATA_TYPE float2
#define MAX_DATA DATA_TYPE(0x0FFFFFFF,0x0FFFFFFF);

//--------------------------------------------------------------------------------------
// Constant Buffers
//--------------------------------------------------------------------------------------
cbuffer CB_Split : register( b0 )
{
	uint NumElements;
	uint NumRegionsH;
	uint NumRegionsV;
	uint SplitOffset;
	uint RegionIndex;
	uint SetMax;
};

struct RegionInfo {

	// the offset of the vertex
	uint VertexOffset;
	
	// the number of the vertex
	uint VertexNum;

};
RWStructuredBuffer<DATA_TYPE> Input;
RWStructuredBuffer<RegionInfo> RegionInfoInput;
RWStructuredBuffer<RegionInfo> RegionInfoOutput;
RWStructuredBuffer<DATA_TYPE> Data;

//--------------------------------------------------------------------------------------
// Find Merge Index for horizontal
//--------------------------------------------------------------------------------------

[numthreads(THREAD_NUM, 1, 1)]
void FindSplitIndexH(uint3 threadId : SV_DispatchThreadID)
{
	if(threadId.x >= NumRegionsH)
		return;
	
	RegionInfo r = (RegionInfo)0;// RegionInfoOutput[threadId.x];
	
	if(threadId.x == 0)
	{
		r.VertexOffset = 0;
		RegionInfoOutput[threadId.x] = r;
		return;
	}
	
	// find the start point of the split
	uint index = threadId.x*SplitOffset;
		
	DATA_TYPE data = Input[index];
	
	// find the bigger index that we have the same y
	[allow_uav_condition] [loop]
	while(	index<NumElements 
			&& (index+1)<NumElements 
			&& Input[index+1].y  == data.y)
	{
		index++;
		data = Input[index];
	}
	
	// update the index
	r.VertexOffset = index;
	RegionInfoOutput[threadId.x] = r;
}

//--------------------------------------------------------------------------------------
// Find Merge Index for Vertical
//--------------------------------------------------------------------------------------

// X vertcal regions , Y horizontal Regions
[numthreads(THREAD_NUM, THREAD_NUM_V_Y, 1)]
void FindSplitIndexV(uint3 threadId : SV_DispatchThreadID)
{
	if(threadId.x >= NumRegionsV)
		return;
	
	if(threadId.y >= NumRegionsH)
		return;
	
	RegionInfo HRegion = RegionInfoInput[threadId.y];

	// find the start point of the split
	uint index = HRegion.VertexOffset + threadId.x * SplitOffset;
	uint maxNum = HRegion.VertexOffset + HRegion.VertexNum;
	
	DATA_TYPE data = Input[index];
	
	// find the bigger index that we have the same y
	[allow_uav_condition] [loop]
	while(	index<maxNum 
			&& (index+1)<maxNum 
			&& Input[index+1].x  == data.x)
	{
		index++;
		data = Input[index];
	}
	
	// update the index
	RegionInfoOutput[ threadId.y * NumRegionsV + threadId.x].VertexOffset = index;
}

//--------------------------------------------------------------------------------------
// Fill the region threads
//--------------------------------------------------------------------------------------


[numthreads(THREAD_NUM, 1, 1)]
void FillRegionInfo(uint3 threadId : SV_DispatchThreadID)
{
	if(threadId.x >= NumRegionsH)
		return;

	RegionInfo r = RegionInfoInput[threadId.x];
	
	// check if we are in the end of the region
	[branch]
	if(threadId.x == NumRegionsH - 1)
	{
		// update the index
		r.VertexNum = NumElements - r.VertexOffset;
	}else{
		// update the index
		r.VertexNum = RegionInfoInput[threadId.x + 1].VertexOffset - r.VertexOffset;
	}
	
	RegionInfoOutput[threadId.x] = r;
}


//--------------------------------------------------------------------------------------
// CopyBuffer Compute Shader
//--------------------------------------------------------------------------------------

[numthreads(1024, 1, 1)]
void CopyRegion( uint3 threadId : SV_DispatchThreadID)
{
	if(threadId.x>=NumElements)
		return;
	
	RegionInfo Region = RegionInfoInput[RegionIndex];

	// startCopy
	[branch]
	if(SetMax == (uint)1)
	{
		[branch]
		if(threadId.x < Region.VertexNum)
			Data[threadId.x] = Input[threadId.x + Region.VertexOffset];
		else 
			Data[threadId.x] = MAX_DATA;
	}else{
		if(threadId.x < Region.VertexNum)
			Data[threadId.x + Region.VertexOffset] = Input[threadId.x];
	}
}

[numthreads(1024, 1, 1)]
void CopySubBuffer( uint3 threadId : SV_DispatchThreadID)
{
	if(threadId.x>=NumElements)
		return;
	
	// startCopy
	[branch]
	if(SetMax == (uint)1)
	{
		[branch]
		if(threadId.x < NumRegionsH)
			Data[threadId.x] = Input[threadId.x + SplitOffset];
		else 
			Data[threadId.x] = MAX_DATA;
	}else{
		if(threadId.x < NumRegionsH)
			Data[threadId.x + SplitOffset] = Input[threadId.x];
	}
}
