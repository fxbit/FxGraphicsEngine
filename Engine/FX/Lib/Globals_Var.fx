//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------

///////////////////////////////////////////////////////////////////////////////////
/// Material
///////////////////////////////////////////////////////////////////////////////////

cbuffer cbMaterial {

// Material's ambient color
float3 g_vMaterialAmbient : Ambient = {0.2f, 0.2f, 0.2f};

// Material's diffuse color
float3 g_vMaterialDiffuse : Diffuse = {0.5f, 0.5f, 1.0f};

// Material's specular color
float3 g_vMaterialSpecular : Specular = {1.0f, 1.0f, 1.0f};

// Transparency of the material
float  g_fMaterialAlpha  = 1.0f;

// How shiny the material is 
float g_nMaterialShininess = 10;

float Ks  = 0.3;

float Eccentricity  = 0.2;

float Kr = 0.7;

} // cbMaterial

///////////////////////////////////////////////////////////////////////////////////
// Lights
///////////////////////////////////////////////////////////////////////////////////

// light color
float3 g_vLightColor : Specular <
	string UIName =  "Lamp 0";
	string Object = "Pointlight0";
	string UIWidget = "Color";
> = {1.0f, 1.0f, 1.0f};    

// light position
float3 g_vLightPosition : POSITION <
	string Object = "PointLight0";
	string UIName =  "Lamp 0 Position";
	string Space = "World";
> = {100.0f,50.0f,100.0f};

	
///////////////////////////////////////////////////////////////////////////////////
// Textures
///////////////////////////////////////////////////////////////////////////////////

// Color texture for mesh
Texture2D g_TextureDiffuse: Diffuse <
	string UIName =  "Color Texture";
	string ResourceType = "2D";
>;

// Normal texture for mesh
Texture2D g_TextureNormal: Diffuse <
	string UIName =  "Normal Texture";
	string ResourceType = "2D";
>;

// Normal texture for mesh
Texture2D g_TextureLightmap: Diffuse <
	string UIName =  "LightMap Texture";
	string ResourceType = "2D";
>;


// Bump texture for mesh
Texture2D g_TextureBump: Diffuse <
	string UIName =  "Bump Texture";
	string ResourceType = "2D";
>;

// highMap texture for mesh
Texture2D g_TextureHighmap: Diffuse <
	string UIName =  "high map Texture";
	string ResourceType = "2D";
>;

//--------------------------------------------------------------------------------------
// Environment
//--------------------------------------------------------------------------------------
TextureCube g_TextureEnvironment : ENVIRONMENT <
	string ResourceName = "CubeMap.dds";
	string UIName =  "Environment";
	string ResourceType = "Cube";
>;

SamplerState EnvSampler = sampler_state {
	//FILTER = MIN_MAG_MIP_LINEAR;
	Filter = ANISOTROPIC;
	MaxAnisotropy = 16;
	AddressU = Clamp;
	AddressV = Clamp;
	AddressW = Clamp;
};

///////////////////////////////////////////////////////////////////////////////////
// Matrix / Camera
///////////////////////////////////////////////////////////////////////////////////

cbuffer cbViewMatrix{

	// transform object vertices to world-space:
	matrix g_mWorld : World;

	// transform object vertices to view space and project them in perspective:
	// matrix with World*View*Projection
	matrix g_mWorldViewProjection : WorldViewProjection;

	// matrix with World* View 
	matrix g_mWorldView : WorldView;

	// matrix with View 
	matrix g_mView : View;

	// matrix with inverse world trans
	matrix g_mWorldInverseTrans : WorldInverseTranspose;

	// inverse view matrix
	matrix g_mViewInverse : ViewInverse;

	// matrix with Projection
	matrix g_mProjection : Projection;

	// matrix with Projection
	matrix g_mViewProjection : Projection;

	// the position of the camera 
	float3 g_vCameraPosition  = { 10.0f,10.0f,10.0f};
};
