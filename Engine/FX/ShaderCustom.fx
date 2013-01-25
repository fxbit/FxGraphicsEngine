

#include "Lib/Globals_Var.fx"
#include "Lib/Structure.fx"
#include "Lib/PerlinNoise.fx"
#include "Lib/CheckerBoard.fx"
#include "Lib/Brick.fx"

//--------------------------------------------------------------------------------------
// Custom Variables
//--------------------------------------------------------------------------------------

cbuffer CustomVariables{

	float3 B1_Color;
	float3 B2_Color;

	VARIABLES_PERLIN_NOISE(Perlin1);
	VARIABLES_PERLIN_NOISE(Perlin2);
	VARIABLES_PERLIN_NOISE(Perlin3);

	VARIABLES_CHECKER(Checker1);

	VARIABLES_BRICK(Brick1);
}


Texture1D <int> Perlin1_RandomTex;
Texture1D <int> Perlin2_RandomTex;
Texture1D <int> Perlin3_RandomTex;
/**/
/*
int[256] Perlin1_RandomTex;
int[256] Perlin2_RandomTex;
int[256] Perlin3_RandomTex; 
*/
Texture2D Tex1_Texture;

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
    float4 vPosProj 		: SV_POSITION;
    float2 UV   			: TEXCOORD0;
	float3 normal			: TEXCOORD1;
	float3 tangent			: TEXCOORD2;
    float3 binormal			: TEXCOORD3;
	float3 vPosWorld 		: TEXCOORD4;
	float3 vPosWorldView 	: TEXCOORD5;
	float3 vLight			: TEXCOORD6;
	float3 vWorldView		: TEXCOORD7;
};

PS_INPUT VS_Main( VS_INPUT input )
{
    PS_INPUT output = (PS_INPUT)0;
    
    // Transform the position into world space for lighting, and projected space
    // for display
    output.vPosProj =  mul( float4(input.PosObject,1), g_mWorldViewProjection );

	// compute the position in the world
    output.vPosWorld =  mul( float4(input.PosObject,1),g_mWorld);
	
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

	// calc v World View
	float4 PosObjectHom = float4(input.PosObject.xyz,1);
	float3 Pw = mul(PosObjectHom,g_mWorld).xyz;
	output.vWorldView = normalize(g_mViewInverse[3].xyz - Pw);
	
	// translate the light position in the world view
	output.vLight = mul(g_vLightPosition.xyz,g_mWorldViewProjection);
    return output;
}

float4 PS_Main( PS_INPUT input) : SV_Target
{
	float4 output;
	
	float noise1 = 0;
	float noise2 = 0;
	
	float weight = 1 ;

	PERLIN_NOISE(Perlin1,input.UV,noise1); 
	
	//////////////////////
	
	PERLIN_NOISE_MULTILEVEL_ABS(Perlin2,input.UV,noise2);

	//////////////////////
	
	PERLIN_NOISE_MULTILEVEL_MARBLE_X(Perlin2,input.UV,noise2);

	//////////////////////
	
	PERLIN_NOISE_MULTILEVEL_MARBLE_Y(Perlin2,input.UV,noise2);

	//////////////////////
	
	PERLIN_NOISE_MULTILEVEL(Perlin2,input.UV,noise2);

	//output.rgb =  noise1 * B1_Color + ( 1 - noise1) * B2_Color;
	/*output.rgb = CheckerBoard(  (int)(input.UV.x*Checker1_CheckerX), 
								(int)(input.UV.y*Checker1_CheckerY),
								noise1 * B1_Color,
								( 1 - noise1) * B2_Color);
								*/

float brick = BRICK( Brick1, input.UV, noise2);

output.rgb = brick * (noise1 * B1_Color + ( 1 - noise1) * B2_Color);

	output.a= 1;
		
    return output;
}

//--------------------------------------------------------------------------------------
// Techniques
//--------------------------------------------------------------------------------------

VertexShader VS_C = CompileShader(vs_5_0, VS_Main());
PixelShader PS_C = CompileShader(ps_5_0, PS_Main());


technique11 Texture
{
    pass
	{
        SetVertexShader( VS_C );
        SetGeometryShader( NULL );
        SetPixelShader( PS_C);
		
			//SetRasterizerState(DisableCulling);
			//SetDepthStencilState(RenderWithStencilState, 0);
		    //SetBlendState(SrcAlphaBlendingAdd, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
    }
}