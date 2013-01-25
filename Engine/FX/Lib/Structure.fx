RasterizerState DisableCulling { CullMode = NONE; };

///////////////////////////////////////////////////////////////////
// Blend States for Alpha blending
///////////////////////////////////////////////////////////////////

BlendState SrcAlphaBlendingAdd
{
   BlendEnable[0]           = TRUE;
   SrcBlend                 = SRC_ALPHA;
   DestBlend                = SRC_ALPHA;
   BlendOp                  = ADD;
   SrcBlendAlpha            = ONE;
   DestBlendAlpha           = ONE;
   BlendOpAlpha             = ADD;
   RenderTargetWriteMask[0] = 0x0F;
};

///////////////////////////////////////////////////////////////////
// Depth/Stencil States
///////////////////////////////////////////////////////////////////

BlendState AlphaBlending
{
    AlphaToCoverageEnable = FALSE;
    BlendEnable[0] = TRUE;
    SrcBlend = SRC_ALPHA;
    DestBlend = INV_SRC_ALPHA;
    BlendOp = ADD;
    SrcBlendAlpha = SRC_ALPHA;
    DestBlendAlpha = INV_SRC_ALPHA;
    BlendOpAlpha = ADD;
    RenderTargetWriteMask[0] = 0x0F;
}; 

BlendState NoBlending
{
    AlphaToCoverageEnable = FALSE;
    BlendEnable[0] = FALSE;
    RenderTargetWriteMask[0] = 0x0F;
};

BlendState SubtractiveBlending
{
    AlphaToCoverageEnable = FALSE;
    BlendEnable[0] = TRUE;
    SrcBlend = ONE;
    DestBlend = ZERO;
    BlendOp = REV_SUBTRACT;         // DST - SRC
    SrcBlendAlpha = ONE;
    DestBlendAlpha = ONE;
    BlendOpAlpha = REV_SUBTRACT;    // DST - SRC
    RenderTargetWriteMask[0] = 0x0F;
};

RasterizerState CullNone
{
    MultiSampleEnable = False;
    CullMode = None;
};

RasterizerState CullFront
{
    MultiSampleEnable = False;
    CullMode = Front;
};

RasterizerState CullBack
{
    MultiSampleEnable = False;
    CullMode = Back;
};


DepthStencilState DepthEnabling { DepthEnable = TRUE; };
DepthStencilState DepthDisabling {
	DepthEnable = FALSE;
	DepthWriteMask = ZERO;
};

DepthStencilState RenderWithStencilState
{
    DepthEnable = TRUE;
    DepthWriteMask = ALL;
    DepthFunc = LESS_EQUAL;
    
    // Setup stencil states
    StencilEnable = false;
    StencilReadMask = 0xFF;
    StencilWriteMask = 0x00;
};

//--------------------------------------------------------------------------------------
// Pipeline State definitions
//--------------------------------------------------------------------------------------
SamplerState samPointClamp
{
    Filter = MIN_MAG_MIP_POINT;
    AddressU = Clamp;
    AddressV = Clamp;
    AddressW = Clamp;
};             

SamplerState samLinearClamp
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Clamp;
    AddressV = Clamp;
    AddressW = Clamp;
};

SamplerState samAniClamp
{
    Filter = ANISOTROPIC;
    AddressU = Clamp;
    AddressV = Clamp;
    AddressW = Clamp;
};
 
SamplerState samRepeat
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Wrap;
    AddressV = Wrap;
};

SamplerState samLinear
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Wrap;
    AddressV = Wrap;
};

SamplerState samLinear1D
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Wrap;
};

SamplerState samPointWrap1D
{
    Filter = MIN_MAG_LINEAR_MIP_POINT;
    AddressU = Wrap;
};       