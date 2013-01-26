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


namespace GraphicsEngine.Core.Shaders.Modules {
    public class ShaderVariable_Checkerboard : ShaderVariable_Base {

        private int _NumBoxX=2;
        private int _NumBoxY=2;

        private FXVariable<int> NumBoxX_Variable;
        private FXVariable<int> NumBoxY_Variable;

        public int NumBoxX
        {
            set { _NumBoxX = value; NumBoxX_Variable.Set(_NumBoxX); }
            get { return _NumBoxX; }
        }

        public int NumBoxY
        {
            set { _NumBoxY = value; NumBoxY_Variable.Set(_NumBoxY); }
            get { return _NumBoxY; }
        }

        public ShaderVariable_Checkerboard(String VariableName)
        {
            this.VariableName = VariableName;
        }

        public override void Init( FXEffect m_effect, FXConstantBuffer m_cb )
        {
            // bind the local variables with the shader
            NumBoxX_Variable = m_cb.GetMemberByName<int>( VariableName + "_CheckerX" );
            NumBoxY_Variable = m_cb.GetMemberByName<int>( VariableName + "_CheckerY" );

            // set the default values
            NumBoxX_Variable.Set(_NumBoxX);
            NumBoxY_Variable.Set(_NumBoxY);
        }

    }
}
