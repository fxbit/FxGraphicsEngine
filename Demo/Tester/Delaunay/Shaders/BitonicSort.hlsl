//--------------------------------------------------------------------------------------
// File: ComputeShaderSort11.hlsl
//
// This file contains the compute shaders to perform GPU sorting using DirectX 11.
// 
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#define BITONIC_BLOCK_SIZE 1024
//#define BITONIC_BLOCK_SIZE 128


#define TRANSPOSE_BLOCK_SIZE 16
//#define TRANSPOSE_BLOCK_SIZE 8

#define DATA_TYPE float2

//--------------------------------------------------------------------------------------
// Constant Buffers
//--------------------------------------------------------------------------------------
cbuffer CB : register( b0 )
{
    uint g_iLevel;
    uint g_iLevelMask;
    uint g_iWidth;
    uint g_iHeight;
	uint g_iField;
};

//--------------------------------------------------------------------------------------
// Structured Buffers
//--------------------------------------------------------------------------------------
RWStructuredBuffer<DATA_TYPE> Input : register( t0 );
RWStructuredBuffer<DATA_TYPE> Data : register( u0 );

//--------------------------------------------------------------------------------------
// Bitonic Sort Compute Shader
//--------------------------------------------------------------------------------------
groupshared DATA_TYPE shared_data[BITONIC_BLOCK_SIZE];

float getSharedDataField(uint index)
{
	[flatten]switch(g_iField)
	{
		case 0:
			return shared_data[index].x;
		case 1:
			return shared_data[index].y;
		//case 3:
			//return shared_data[index].z;
	}
	
	return -1;
}

[numthreads(BITONIC_BLOCK_SIZE, 1, 1)]
void BitonicSort( uint3 DTid : SV_DispatchThreadID,
				  uint3 GTid : SV_GroupThreadID, 
				  uint3 Gid : SV_GroupID, 
                  uint GI : SV_GroupIndex )
{		
    // Load shared data
	shared_data[GI] = Data[DTid.x];
	
    GroupMemoryBarrierWithGroupSync();
    
    // Sort the shared data
    for (unsigned int j = g_iLevel >> 1 ; j > 0 ; j >>= 1)
    {
		DATA_TYPE result = ((getSharedDataField(GI & ~j) <= getSharedDataField(GI | j)) == (bool)(g_iLevelMask & DTid.x))? shared_data[GI ^ j] : shared_data[GI];
		GroupMemoryBarrierWithGroupSync();
		shared_data[GI] = result;
		GroupMemoryBarrierWithGroupSync();
    }
    
    // Store shared data
	Data[DTid.x] = shared_data[GI];
}

//--------------------------------------------------------------------------------------
// Matrix Transpose Compute Shader
//--------------------------------------------------------------------------------------
groupshared DATA_TYPE transpose_shared_data[TRANSPOSE_BLOCK_SIZE * TRANSPOSE_BLOCK_SIZE];

[numthreads(TRANSPOSE_BLOCK_SIZE, TRANSPOSE_BLOCK_SIZE, 1)]
void MatrixTranspose( uint3 DTid : SV_DispatchThreadID, 
                      uint3 GTid : SV_GroupThreadID, 
                      uint GI : SV_GroupIndex )
{
	uint Input_X = DTid.y * g_iWidth + DTid.x;
	transpose_shared_data[GI] = Input[Input_X];
	
    GroupMemoryBarrierWithGroupSync();
    
	uint2 XY = DTid.yx - GTid.yx + GTid.xy;
	uint Data_X = XY.y * g_iHeight + XY.x;
	uint Share_X =  GTid.x * TRANSPOSE_BLOCK_SIZE + GTid.y;
	Data[Data_X] = transpose_shared_data[Share_X];
}

//--------------------------------------------------------------------------------------
// Main Dummy Compute Shader
//--------------------------------------------------------------------------------------
[numthreads(TRANSPOSE_BLOCK_SIZE, TRANSPOSE_BLOCK_SIZE, 1)]
void main( uint3 DTid : SV_DispatchThreadID, 
           uint3 GTid : SV_GroupThreadID, 
           uint GI : SV_GroupIndex )
{
	
	
}
