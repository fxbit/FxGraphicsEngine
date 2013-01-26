using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

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
    public class ShaderVariable_Base {

        /// <summary>
        /// the name of the variable that we want
        /// </summary>
        internal String VariableName = "";

        public virtual void Init( FXEffect m_effect, FXConstantBuffer m_cb )
        {

        }

        public override string ToString()
        {
            return VariableName;
        }
    }
}
