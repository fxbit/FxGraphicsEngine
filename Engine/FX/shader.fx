

#include "Lib/Globals_Var.fx"
#include "Lib/Structure.fx"
#define FLT_MAX 1E+37

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
    float4 vColor   		: COLOR0;
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
    float4 position = float4(input.PosObject,1);
	
	position.y += g_TextureHighmap.SampleLevel( samAniClamp, input.UV ,0 ).x;
	
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

    // calc v World View
    float4 PosObjectHom = float4(position.xyz,1);
    float3 Pw = mul(PosObjectHom,g_mWorld).xyz;
    output.vWorldView = normalize(g_mViewInverse[3].xyz - Pw);
    
    // translate the light position in the world view
    output.vLight = mul(g_vLightPosition.xyz,g_mWorldViewProjection);
    return output;
}

void blinn_shading(
            PS_INPUT input,
            float3 LightColor,
            float3 Nn,
            float3 Ln,
            float3 Vn,
            out float3 DiffuseContrib,
            out float3 SpecularContrib)
{
    float3 Hn = normalize(Vn + Ln);
    float hdn = dot(Hn,Nn);
    float3 R = reflect(-Ln,Nn);
    float rdv = saturate(dot(R,Vn) + 0.001) ;
    float ldn = saturate(dot(Ln,Nn)) ;
    float ndv = dot(Nn,Vn);
    float hdv = dot(Hn,Vn);
    float eSq = Eccentricity*Eccentricity;
    float distrib = eSq / (rdv * rdv * (eSq - 1.0) + 1.0);
    distrib = distrib * distrib;
    float Gb = 2.0 * hdn * ndv / hdv;
    float Gc = 2.0 * hdn * ldn / hdv;
    float Ga = min(1.0,min(Gb,Gc));
    float fresnelHack = 1.0 - pow(ndv,5.0f);
    hdn = distrib * Ga * fresnelHack / ndv;
    DiffuseContrib = pow(ldn,2) * LightColor;
    SpecularContrib = hdn * Ks * LightColor ;
}

float4 PS_Main( PS_INPUT input) : SV_Target
{

    float4 output;
    
    // view and light directions
    float3 Vn = normalize(input.vPosProj);
    float3 Ln = normalize(g_vLightPosition-input.vPosWorldView);
    
    float3 tNorm = input.normal;
    
    float3 texCol = g_TextureDiffuse.Sample( samAniClamp, input.UV );
    
	float3 light = g_TextureLightmap.Sample( samAniClamp, input.UV );
	
    // Compute the ambient and diffuse components of illumination
    output.rgb = texCol*light;
	
    /*
    float att = saturate(dot(Ln,input.normal));
    float diff = saturate(dot(Ln,tNorm));
    output.rgb += texCol*g_vLightColor * g_vMaterialDiffuse * att*diff;
    
    // specular terms
    float spec = saturate(dot(normalize(Ln-Vn),tNorm));
    spec = pow(spec,g_nMaterialShininess);
    output.rgb +=g_vMaterialSpecular*spec*att;
    */
    output.a= g_fMaterialAlpha;
        
    return output;
}

float4 PS_NOTEX( PS_INPUT input) : SV_Target
{

    float4 output;
    
    // view and light directions
    float3 Vn = normalize(input.vPosProj);
    float3 Ln = normalize(g_vLightPosition-input.vPosWorldView);
    
    float3 tNorm = input.normal;
    
    // Compute the ambient and diffuse components of illumination
    output.rgb = g_vMaterialDiffuse*g_vLightColor * g_vMaterialAmbient;
    
    float att = saturate(dot(Ln,input.normal));
    float diff = saturate(dot(Ln,tNorm));
    output.rgb += g_vMaterialDiffuse*g_vLightColor * g_vMaterialDiffuse * att*diff;
    /*
    // specular terms
    float spec = saturate(dot(normalize(Ln-Vn),tNorm));
    spec = pow(spec,g_nMaterialShininess);
    output.rgb +=g_vMaterialSpecular*spec*att;
    */
    output.a= g_fMaterialAlpha;
        
    return output;
}

float4 PS_SMOKE( PS_INPUT input) : SV_Target
{

    float4 output;

    float3 texCol = g_TextureDiffuse.Sample( samLinear, input.UV );

    output.rgb =texCol/255;
    output.a= 1;
        
    return output;
}

float4 PS_BUMP( PS_INPUT input) : SV_Target
{
    float4 output;
    
    float3 tNorm = g_TextureNormal.Sample(samLinear,input.UV) - float3(0.5,0.5,0.5);
    
    // transform tNorm to world space
    tNorm = normalize(tNorm.x*input.tangent -
              tNorm.y*input.binormal + 
              tNorm.z*input.normal);
              
    float3 texCol = g_TextureDiffuse.Sample( samLinear, input.UV );
    
    // view and light directions
    float3 Vn = normalize(input.vPosWorldView);
    float3 Ln = normalize(g_vLightPosition-input.vPosWorldView);
    
    // Compute the ambient and diffuse components of illumination
    output.rgb = texCol*g_vLightColor * g_vMaterialAmbient;
    float att = saturate(dot(Ln,input.normal));
    float diff = saturate(dot(Ln,tNorm));
    output.rgb += texCol*g_vLightColor * g_vMaterialDiffuse * att*diff;
    
    // specular terms
    float spec = saturate(dot(normalize(Ln-Vn),tNorm));
    spec = pow(spec,g_nMaterialShininess);
    output.rgb +=g_vMaterialSpecular*spec*att;
        
    output.a=g_fMaterialAlpha;
    return output;
}
    

float4 PS_Relief( PS_INPUT input) : SV_Target
{
    float3 result = (float3)0;
    
    /// normalize the input
    float3 Ln = normalize(input.vLight);
    float3 Nn = normalize(input.normal);
    float3 Tn = normalize(input.tangent);
    float3 Bn = normalize(input.binormal);
    float3 Vn = normalize(input.vWorldView);
    
    float3 diffContrib;
    float3 specContrib ;
    
    blinn_shading(input,g_vLightColor,Nn,Ln,Vn,diffContrib,specContrib);
    result = specContrib + g_vMaterialAmbient + g_vMaterialDiffuse * diffContrib;
    return float4(result,1);
}
