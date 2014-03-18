
#include "Lib/Globals_Var.fx"
#include "Lib/Structure.fx"
#define FLT_MAX 1E+37

#define quadLength 0.1f

struct Particle{
	float3 pos    	: POSITION;
	float3 color	: COLOR;
};

StructuredBuffer<Particle> particleBuffer;

//--------------------------------------------------------------------------------------
// Pixel shader input structure
//--------------------------------------------------------------------------------------
struct PS_INPUT
{
	float4	pos		: SV_POSITION;
	float2  tex0	: TEXCOORD0;
	float3	color	: COLOR;
};

//--------------------------------------------------------------------------------------
// Geometry shader input structure
//--------------------------------------------------------------------------------------
struct GS_INPUT
{
	float4	pos		: SV_POSITION;
	float2  tex0	: TEXCOORD0;
	float3	color	: COLOR;
};


//--------------------------------------------------------------------------------------
// Pixel shader main Function
//--------------------------------------------------------------------------------------
GS_INPUT VS_Main(uint vid : SV_VertexID)
{
	GS_INPUT output = (GS_INPUT)0;

	output.pos = float4(particleBuffer[vid].pos, 1);
	output.color = particleBuffer[vid].color;

	return output;
}

//--------------------------------------------------------------------------------------
// Pixel shader main Function
//--------------------------------------------------------------------------------------
float4 PS_Main(PS_INPUT input) : SV_TARGET
{
	float4 output = g_TextureDiffuse.Sample(samAniClamp, input.tex0);
	//output.rgb = output.rgb * input.color;
	output.rgb = input.color;
	output.a = 1.0f;
	return output;
}


//--------------------------------------------------------------------------------------
// Geometry shader main
//--------------------------------------------------------------------------------------
[maxvertexcount(4)]
void GS_Main(point GS_INPUT p[1], inout TriangleStream<PS_INPUT> triStream)
{
#if 0
	PS_INPUT p1 = (PS_INPUT)0;
	float3 edge;

	float3 normal = p[0].pos.xyz - g_vCameraPosition;
	normal = mul(float4(normal, 1), g_mView).xyz;

	float3 rightAxis = cross(float3(0.0f, 1.0f, 0.0f), normal);
	float3 upAxis = cross(normal, rightAxis);

	float3 rightVector = normalize(rightAxis);
	float3 upVector = normalize(upAxis);
	float3 center = mul(float4(p[0].pos, 1), g_mView).xyz;

	edge = center + rightVector*(quadLength)+upVector*(quadLength);
	p1.tex0 = float2(1.0f, 0.0f);
	p1.pos = mul(float4(edge, 1), g_mProjection);
	p1.color = p[0].color;
	triStream.Append(p1);

	edge = center + rightVector*(quadLength)+upVector*(-quadLength);
	p1.tex0 = float2(1.0f, 1.0f);
	p1.pos = mul(float4(edge, 1), g_mProjection);
	p1.color = p[0].color;
	triStream.Append(p1);

	edge = center + rightVector*(-quadLength) + upVector*(quadLength);
	p1.tex0 = float2(0.0f, 0.0f);
	p1.pos = mul(float4(edge, 1), g_mProjection);
	p1.color = p[0].color;
	triStream.Append(p1);

	edge = center + rightVector*(-quadLength) + upVector*(-quadLength);
	p1.tex0 = float2(0.0f, 1.0f);
	p1.pos = mul(float4(edge, 1), g_mProjection);
	p1.color = p[0].color;
	triStream.Append(p1);

#else

	PS_INPUT p1;
	float3 up = float3(0.0f, 1.0f, 0.0f);
	float3 look = p[0].pos.xyz - g_vCameraPosition;
	look = normalize(look);
	float3 right = normalize(cross(up, look));
	up = normalize(cross(look, right));

	float3 halfWidth = 0.5f * quadLength * right;
	float3 halfHeight = 0.5f * quadLength * up;

	// point 1
	p1.pos = mul(float4(p[0].pos + halfWidth - halfHeight, 1.0f), g_mViewProjection);
	p1.tex0 = float2(0.0f, 1.0f);
	p1.color = p[0].color;
	triStream.Append(p1);

	// point 2
	p1.pos = mul(float4(p[0].pos + halfWidth + halfHeight, 1.0f), g_mViewProjection);
	p1.tex0 = float2(0.0f, 0.0f);
	p1.color = p[0].color;
	triStream.Append(p1);

	// point 3
	p1.pos = mul(float4(p[0].pos - halfWidth - halfHeight, 1.0f), g_mViewProjection);
	p1.tex0 = float2(1.0f, 1.0f);
	p1.color = p[0].color;
	triStream.Append(p1);

	// point 4
	p1.pos = mul(float4(p[0].pos - halfWidth + halfHeight, 1.0f), g_mViewProjection);
	p1.tex0 = float2(1.0f, 0.0f);
	p1.color = p[0].color;
	triStream.Append(p1);

#endif
}