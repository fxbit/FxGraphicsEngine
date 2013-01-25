

#include "Lib/Globals_Var.fx"
#include "Lib/Structure.fx"
#define FLT_MAX 1E+37

cbuffer cbPerlinNoiseVariables{
	float MinValue;
	float MaxValue;
	float Tile_X;
	float Tile_Y;
	int Loops;
}

// Normal texture for mesh
Texture1D <int> randomTex;

//--------------------------------------------------------------------------------------
// Vertex shader input structure
//--------------------------------------------------------------------------------------
struct VS_INPUT
{
	float3 PosObject	: POSITION;
    float3 normal		: NORMAL; 		// expected to be normalized
    float2 UV			: TEXCOORD0;
	float3 tangent		: TANGENT0; 	// pre-normalized
    float3 binormal		: BINORMAL0; 	// pre-normalized
};

//--------------------------------------------------------------------------------------
// Pixel shader input structure
//--------------------------------------------------------------------------------------
struct PS_INPUT
{
    float4 vPosProj : SV_POSITION;
    float4 vColor   : COLOR0;
    float2 UV   	: TEXCOORD0;
	float3 normal	: TEXCOORD1;
	float3 tangent	: TEXCOORD2;
    float3 binormal	: TEXCOORD3;
	float3 vPosWorld : TEXCOORD4;
	float3 vPosWorldView : TEXCOORD5;
};

PS_INPUT VS_Main( VS_INPUT input )
{
    PS_INPUT output;
    
    // Transform the position into world space for lighting, and projected space
    // for display
    output.vPosProj =  mul( float4(input.PosObject,1), g_mWorldViewProjection );

	// compute the position in the world
    output.vPosWorld =  mul( float4(input.PosObject,1),g_mWorld).rgb;
	
	// compute the position in the world
    output.vPosWorldView =  mul( float4(input.PosObject,1),g_mWorldView);
	
    // Pass the texture coordinate // and inverse
    output.UV =input.UV;

	// staff for bump 
	
	// isolate WorldView rotation-only part
    float3x3 modelViewRot;
    modelViewRot[0] = g_mWorldView[0].xyz;
    modelViewRot[1] = g_mWorldView[1].xyz;
    modelViewRot[2] = g_mWorldView[2].xyz;
	
    // tangent space vectors in view space (with model transformations)
    output.tangent 	= normalize(mul(input.tangent,modelViewRot));
    output.binormal = normalize(mul(input.binormal,modelViewRot));
    output.normal 	= normalize(mul(input.normal,modelViewRot));

    return output;
}

/// Smooth the entry value
/// <param name="t">The entry value</param>
/// <returns>The smoothed value</returns>
float Fade(float t)
{
    return t * t * t * (t * (t * 6 - 15) + 10);
}

/// Modifies the result by adding a directional bias
/// <param name="hash">The random value telling in which direction the bias will occur</param>
/// <param name="x">The amount of the bias on the X axis</param>
/// <param name="y">The amount of the bias on the Y axis</param>
/// <returns>The directional bias strength</returns>
float Grad(int hash, float x, float y)
{
    // Fetch the last 3 bits
    int h = hash & 3;

    // Result table for U
    // ---+------+---+------
    //  0 | 0000 | x |  x
    //  1 | 0001 | x |  x
    //  2 | 0010 | x | -x
    //  3 | 0011 | x | -x

    float u = (h & 2) == 0 ? x : -x;

    // Result table for V
    // ---+------+---+------
    //  0 | 0000 | y |  y
    //  1 | 0001 | y | -y
    //  2 | 0010 | y |  y
    //  3 | 0011 | y | -y

    float v = (h & 1) == 0 ? y : -y;

    // Result table for U + V
    // ---+------+----+----+--
    //  0 | 0000 |  x |  y |  x + y
    //  1 | 0001 |  x | -y |  x - y
    //  2 | 0010 | -x |  y | -x + y
    //  3 | 0011 | -x | -y | -x - y

    return u + v;
}

/// Generates a bi-dimensional noise
float Noise(float x, float y, Texture1D <int> RandTex)
{
	// Compute the cell coordinates
	int X = (int)floor(x) & 255;
	int Y = (int)floor(y) & 255;

	// Retrieve the decimal part of the cell
	x -= floor(x);
	y -= floor(y);

	// Smooth the curve
	float u = Fade(x);
	float v = Fade(y);

	// Fetch some random values from the table
	int A = RandTex.Load( int2(X		,0)) + Y;
	int B = RandTex.Load( int2(X + 1	,0)) + Y; 

	// Interpolate between directions 
	return lerp(lerp(Grad(RandTex.Load( int2(A		,0))	, x    , y	  ) ,
					 Grad(RandTex.Load( int2(B		,0))	, x - 1, y	  ) , u),
				lerp(Grad(RandTex.Load( int2(A + 1	,0))	, x    , y - 1) ,
					 Grad(RandTex.Load( int2(B + 1	,0))	, x - 1, y - 1) , u) , v);
}

/// scale the values to the output
float Normalize(float value, float NoiseMinValue,float h){
	return (((value - NoiseMinValue) * h) * (MaxValue - MinValue) + MinValue);
}

//////////////////////////////////////////////////////////////////////////////////////////////
// special type of noise

float PerlinNoise_1(float x,float y){
	return Normalize(Noise(x,y, randomTex),-1,0.5);
}

float PerlinNoise_2(float x,float y){
	float value=0;
	float weight = 1 ;

	for(int i=0;i<Loops;i++)
	{
		value += Noise(weight * x ,weight * y, randomTex) / weight;
		weight*=2;
	}

	return Normalize(value,-1,0.5);
}

float PerlinNoise_3(float x,float y){
	float value=0;
	float weight = 1 ;

	for(int i=0;i<Loops;i++)
	{
		value +=  abs(Noise(weight * x ,weight * y, randomTex) / weight);
		weight*=2;
	}

	return Normalize(value,-1,0.5);
}

float PerlinNoise_4(float x,float y){
	float value=0;
	float weight = 1 ;

	for(int i=0;i<Loops;i++)
	{
		value +=  abs(Noise(weight * x ,weight * y, randomTex) / weight);
		weight*=2;
	}

	value = sin( x + value); 

	return Normalize(value,-1,0.5);
}

//////////////////////////////////////////////////////////////////////////////////////////////


float4 PS_1( PS_INPUT input) : SV_Target
{

	float4 output;

	float value = PerlinNoise_1(input.UV.x*Tile_X,input.UV.y*Tile_Y);

	output.rgb = float3(value,value,value);
	output.a= 0;
		
    return output;
}

float4 PS_Main( PS_INPUT input) : SV_Target
{

	float4 output;
	
	float value = PerlinNoise_2(input.UV.x*Tile_X,input.UV.y*Tile_Y);

	output.rgb = float3(value,value,value);
	output.a= 1;
		
    return output;
}

float4 PS_3( PS_INPUT input) : SV_Target
{

	float4 output;

	float value = PerlinNoise_3(input.UV.x*Tile_X,input.UV.y*Tile_Y);

	output.rgb = float3(value,value,value);
	output.a= 1;
		
    return output;
}

float4 PS_4( PS_INPUT input) : SV_Target
{

	float4 output;

	float value = PerlinNoise_4(input.UV.x*Tile_X,input.UV.y*Tile_Y);

	output.rgb = float3(value,value,value);
	output.a= 1;
		
    return output;
}
