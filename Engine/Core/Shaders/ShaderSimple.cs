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



namespace GraphicsEngine.Core.Shaders {
    public class ShaderSimple : Shader {

        /// <summary>
        /// the name of the noise that we want
        /// </summary>
        const String ShaderName = "FX/shader.fx";

        #region Texture Variables
        protected FXResourceVariable m_TextureDiffuse;
        protected FXResourceVariable m_TextureNormal;
        protected FXResourceVariable m_TextureLightmap;
        protected FXResourceVariable m_TextureHighmap;
        protected FXResourceVariable m_TextureBump;
        #endregion

        #region Constructor
        public ShaderSimple()
            : base(ShaderName)
        {
            InitShader();
        }

        public ShaderSimple(String Name)
            : base(Name)
        {
            InitShader();
        }
        #endregion

        private void InitShader()
        {

            /// Texture resources
            m_TextureDiffuse = m_effect.GetResourceByName("g_TextureDiffuse");
            m_TextureBump = m_effect.GetResourceByName( "g_TextureBump");
            m_TextureNormal = m_effect.GetResourceByName( "g_TextureNormal");
            m_TextureLightmap = m_effect.GetResourceByName("g_TextureLightmap");
            m_TextureHighmap = m_effect.GetResourceByName("g_TextureHighmap");
        }

        /// <summary>
        /// Set variables to the shader
        /// This is for vector
        /// </summary>
        /// <param name="data"></param>
        /// <param name="type"></param>
        public void SetVariables(Vector3 data,  ShaderViariables type)
        {
            switch ( type ) {
                case ShaderViariables.Diffuse: m_Diffuse.Set(data); break;
                case ShaderViariables.Ambient:m_Ambient.Set(data); break;
                case ShaderViariables.Specular: m_Specular.Set(data); break;
            }
        }

        /// <summary>
        /// Set variables to the shader
        /// This is for scalar
        /// </summary>
        /// <param name="data"></param>
        /// <param name="type"></param>
        public void SetVariables(float data, ShaderViariables type)
        {
            switch (type) {
                case ShaderViariables.SpecularPower: m_SpecularPower.Set(data); break;
            }
        }

        /// <summary>
        /// Set the texture that the mesh use
        /// </summary>
        /// <param name="texturePath">Where on disk the texture is saved</param>
        /// <param name="textType"></param>
        public Texture SetTexture(string texturePath, TextureType textType)
        {
            /// add the texture to tha manager and if is not new just get the resource
            Texture tex = TextureManager.AddTexture(texturePath);
            
            switch (textType) {
                case TextureType.Diffuse:
                    m_TextureDiffuse.SetResource(tex.shaderResource); break;
                case TextureType.Bump:
                    m_TextureBump.SetResource(tex.shaderResource); break;
                case TextureType.Normal:
                    m_TextureNormal.SetResource(tex.shaderResource); break;
                case TextureType.Lightmap:
                    m_TextureLightmap.SetResource(tex.shaderResource); break;
                case TextureType.Heightmap:
                    m_TextureHighmap.SetResource(tex.shaderResource); break;
                    
            }

            return tex;
        }

        /// <summary>
        /// Set the texture that the mesh use
        /// </summary>
        /// <param name="texturePath">Where on disk the texture is saved</param>
        /// <param name="textType"></param>
        public void SetTexture(Texture tex, TextureType textType)
        {
            switch (textType)
            {
                case TextureType.Diffuse:
                    m_TextureDiffuse.SetResource(tex.shaderResource); break;
                case TextureType.Bump:
                    m_TextureBump.SetResource(tex.shaderResource); break;
                case TextureType.Normal:
                    m_TextureNormal.SetResource(tex.shaderResource); break;
                case TextureType.Lightmap:
                    m_TextureLightmap.SetResource(tex.shaderResource); break;
                case TextureType.Heightmap:
                    m_TextureHighmap.SetResource(tex.shaderResource); break;

            }
        }

        /// <summary>
        /// Set the texture that the mesh use
        /// </summary>
        /// <param name="texturePath">Where on disk the texture is saved</param>
        /// <param name="textType"></param>
        public FXResourceVariable GetTexture(TextureType textType)
        {
            switch (textType)
            {
                case TextureType.Diffuse:
                   return m_TextureDiffuse;
                case TextureType.Bump:
                   return m_TextureBump;
                case TextureType.Normal:
                   return m_TextureNormal;
                case TextureType.Lightmap:
                   return m_TextureLightmap;
                case TextureType.Heightmap:
                   return m_TextureHighmap;

            }

            return null;
        }
    }
}
