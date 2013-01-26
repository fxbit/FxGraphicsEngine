using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;

using GraphicsEngine.Managers;
using FXFramework;

using SharpDX;
using SharpDX.DXGI;
using SharpDX.Direct3D11;

// resolve conflict - DXGI.Device & Direct3D10.Device
using Device = SharpDX.Direct3D11.Device;
using Buffer = SharpDX.Direct3D11.Buffer;
using Effect = SharpDX.Direct3D11.Effect;
using EffectFlags = SharpDX.D3DCompiler.EffectFlags;
using Color = SharpDX.Color;

namespace GraphicsEngine.Core.Shaders.Modules {
    public class ShaderVariable_Color : ShaderVariable_Base {

        private Vector3 _color;
        private FXVariable<Vector3> Color_Variable;

        public Vector3 Color
        {
            set { _color = value; Color_Variable.Set(_color); }
            get { return _color; }
        }

        public Color Color_C
        {
            set { _color = new Vector3(value.R/255.0f,value.G/255.0f,value.B/255.0f) ; Color_Variable.Set(_color); }
            get { return new Color(255, 
                                    (int)(_color.X * 255), // Red
                                    (int)(_color.Y * 255), // Green
                                    (int)(_color.Z * 255)); }// Blue
        }

        public ShaderVariable_Color(String VariableName)
        {
            this.VariableName = VariableName;
        }

        public override void Init( FXEffect m_effect, FXConstantBuffer m_cb )
        {
            // bind the local variables with the shader
            Color_Variable = m_cb.GetMemberByName<Vector3>( VariableName + "_Color" );
        
            // set the default values
            Color_Variable.Set(_color);
        }

    }
}
