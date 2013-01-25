
//Texture2D Input : register(t0);
//RWTexture2D<float4> Output : register(u0);

/*
uint3 threadIDInGroup : SV_GroupThreadID (ID within the group, in each dimension
uint3 groupID : SV_GroupID, (ID of the group, in each dimension of the dispatch)
uint groupIndex : SV_GroupIndex (flattened ID of the group in one dimension if you counted like a raster)
uint3 dispatchThreadID : SV_DispatchThreadID (ID of the thread within the entire dispatch in each dimension)
*/
[numthreads(1, 1, 1)]
 void main( uint3 threadIDInGroup : SV_GroupThreadID, 
			uint3 groupID : SV_GroupID,
			uint  groupIndex : SV_GroupIndex,     
			uint3 dispatchThreadID : SV_DispatchThreadID )
{
   // Output[dispatchThreadID.xy]= float4(dispatchThreadID.xy / 1024.0f, 0, 1);
}