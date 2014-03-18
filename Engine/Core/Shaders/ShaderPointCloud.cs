using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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

namespace GraphicsEngine.Core.Shaders
{
    public class ShaderPointCloud : Shader
    {

        /// <summary>
        /// the name of the noise that we want
        /// </summary>
        const String ShaderName = "FX/PointCloud.fx";


        #region Texture Variables
        protected FXResourceVariable m_TextureDiffuse;
        #endregion


        #region Constructor
        public ShaderPointCloud()
            : base(ShaderName)
        {
            InitShader();
        }

        public ShaderPointCloud(String Name)
            : base(Name)
        {
            InitShader();
        }
        #endregion


        private void InitShader()
        {
            /// Texture resources
            m_TextureDiffuse = m_effect.GetResourceByName("g_TextureDiffuse");
        }
    }
}
