using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using SharpDX;
using SharpDX.DXGI;
using SharpDX.Direct3D11;

// resolve conflict - DXGI.Device & Direct3D10.Device
using Device = SharpDX.Direct3D11.Device;
using Buffer = SharpDX.Direct3D11.Buffer;
using Effect = SharpDX.Direct3D11.Effect;
using EffectFlags = SharpDX.D3DCompiler.EffectFlags;
using GraphicsEngine.Managers;
using GraphicsEngine.Core.Shaders.Modules;
using FXFramework;

namespace GraphicsEngine.Core.Shaders {

    public class ShaderCustom : Shader {

        /// <summary>
        /// the name of the noise that we want
        /// </summary>
        const String ShaderName = "FX/ShaderCustom.fx";


        List<ShaderVariable_Base> _ListVariables = new List<ShaderVariable_Base>();

        FXConstantBuffer cbCustomShader;

        public List<ShaderVariable_Base> ListVariables
        {
            get { return _ListVariables; }
        }

        #region Constructor
        public ShaderCustom()
            : base(ShaderName)
        {
            InitShader();
        }

        public ShaderCustom(String Name)
            : base(Name)
        {
            InitShader();
        }
        #endregion

        private void InitShader()
        {
            /// get the constant buffer that have all of our variables
            cbCustomShader = m_effect.GetConstantBufferByName( "CustomVariables" );
        }

        public void AddVariables(ShaderVariable_Base var)
        {
            /// add the variables to the list
            _ListVariables.Add(var);

            /// init the variable base on the Shader
            var.Init( m_effect, cbCustomShader );
        }

        /// <summary>
        /// Compile in runtime the shader that 
        /// get from the memmry
        /// </summary>
        /// <param name="Shader"></param>
        /// <returns>The errors from the building</returns>
        public override String RunTimeCompile(String Shader)
        {
            String Result = base.RunTimeCompile(Shader);

            // check if the compile was correct
            if (Result.Equals("")) {

                foreach (ShaderVariable_Base shBase in _ListVariables) {
                //   shBase.Init(m_effect);
                }

            }


            return Result;
        }
    }
}
