
#include "Lib/Globals_Var.fx"
#include "Lib/Structure.fx"
#define FLT_MAX 1E+37

#define quadLength 0.2f

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
	float3	color	: COLOR;
	float2  tex0	: TEXCOORD0;
};

//--------------------------------------------------------------------------------------
// Geometry shader input structure
//--------------------------------------------------------------------------------------
struct GS_INPUT
{
	float4	pos		: SV_POSITION;
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
	float4 output = g_TextureDiffuse.Sample(samPointClamp, input.tex0);
	output.a = output.r;
	output.rgb = output.rgb * input.color;
	return output;
}


//--------------------------------------------------------------------------------------
// Geometry shader main
//--------------------------------------------------------------------------------------
[maxvertexcount(4)]
void GS_Main(point GS_INPUT p[1], inout TriangleStream<PS_INPUT> triStream)
{
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
}