using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

using FXFramework;

// SharpDX library
using SharpDX;
using SharpDX.DXGI;
using SharpDX.Direct3D11;
using SharpDX.D3DCompiler;

// resolve conflict - DXGI.Device & Direct3D10.Device
using Device = SharpDX.Direct3D11.Device;
using Buffer = SharpDX.Direct3D11.Buffer;
using Effect = SharpDX.Direct3D11.Effect;
using EffectFlags = SharpDX.D3DCompiler.EffectFlags;

// internal librarys
using GraphicsEngine.Core.Shaders;
using SharpDX.Direct3D;


namespace GraphicsEngine.Core {
    public class IncludeFX : Include {

        public String RelIncludeDirectory
        {
            get { return includeDirectory; }
            set { includeDirectory =  System.Windows.Forms.Application.StartupPath + "\\" + value; }
        }

        public String IncludeDirectory
        {
            get { return includeDirectory; }
            set { includeDirectory = value; }
        }

        string includeDirectory = System.Windows.Forms.Application.StartupPath + "\\FX\\";

        public void Close(Stream stream)
        {
            stream.Close();
        }

        public void Open(IncludeType type, string fileName, Stream parentStream, out Stream stream)
        {
            stream = new FileStream(includeDirectory + fileName, FileMode.Open, FileAccess.Read, FileShare.Read);
        }


        public Stream Open(IncludeType type, string fileName, Stream parentStream)
        {
            return new FileStream(includeDirectory + fileName, FileMode.Open, FileAccess.Read, FileShare.Read);
        }

        public IDisposable Shadow
        {
            get
            {
                return this;
            }
            set
            {
             //   throw new NotImplementedException();
            }
        }

        public void Dispose()
        {
            throw new NotImplementedException();
        }
    }

    public class Shader {

        String ShaderRootPath = System.Windows.Forms.Application.StartupPath + "\\";

        #region Shader's Matrices


        protected FXConstantBuffer m_ViewConstantBuffer;
        protected FXVariable<Matrix> m_WorldMatrixVariable;
        protected FXVariable<Matrix> m_ViewMatrixVariable;
        protected FXVariable<Matrix> m_WorldViewMatrixVariable;
        protected FXVariable<Matrix> m_WorldViewProjectionMatrixVariable;
        protected FXVariable<Matrix> m_WorldInverseTranspose;
        protected FXVariable<Matrix> m_ViewInverse;

        #endregion

        /// <summary>
        /// The bytecode of the pixel shader
        /// </summary>
        public ShaderBytecode m_PixelShaderByteCode;

        /// <summary>
        /// The bytecode of the vertex shader
        /// </summary>
        public ShaderBytecode m_VertexShaderByteCode;

        /// <summary>
        /// Shader effect
        /// </summary>
        public FXEffect m_effect;

        #region Material Variables

        protected FXConstantBuffer m_MaterialBuffer;
        protected FXVariable<Vector3> m_Ambient;
        protected FXVariable<Vector3> m_Diffuse;
        protected FXVariable<Vector3> m_Specular;
        protected FXVariable<float> m_SpecularPower;
        protected FXVariable<Vector3> m_LightColor;
        protected FXVariable<Vector3> m_LightPosition;
        protected FXVariable<Vector3> m_CameraPosition;
        protected FXVariable<float> m_Opacity;

        #endregion


        public Shader(String Name, String VS_EntryPoint="VS_Main" , String PS_EntryPoint = "PS_Main" ,Boolean PreCompiled=false)
        {
            Device dev = Engine.g_device;

            IncludeFX includeFX = new IncludeFX();
            String errors;

            if ( PreCompiled ) {
                // check that the shader is exist
                if ( File.Exists( ShaderRootPath + Name + "o" ) ) {
                    // open the file to read it 
                    FileStream fileStream = new FileStream( ShaderRootPath + Name + "o", FileMode.Open );

                    // allocate the byte stream
                    byte[] fileByte = new byte[fileStream.Length];

                    // read the file stream
                    fileStream.Read( fileByte, 0, (int)fileStream.Length );

                    // close the file stream
                    fileStream.Close();

                    DataStream preBuildShaderStream = new DataStream( fileByte.Length, true, true );
                    preBuildShaderStream.Write(fileByte, 0, fileByte.Length);

                    m_PixelShaderByteCode = new ShaderBytecode( preBuildShaderStream );
                    m_VertexShaderByteCode = new ShaderBytecode( preBuildShaderStream );
                } else {
                    System.Windows.Forms.MessageBox.Show( "Shader:" + ShaderRootPath + Name + "o" + "   is not exist " );

                    return;
                }

            } else {
                // set the shader flags base on the debugging
                ShaderFlags sf;
                if ( Settings.Debug ) {
                    sf = ShaderFlags.SkipOptimization | ShaderFlags.Debug | ShaderFlags.PreferFlowControl;
                } else {
                    sf = ShaderFlags.OptimizationLevel3;
                }


                // set the compile feature
                String CompileLevelPS;
                String CompileLevelVS;

                if (Settings.FeatureLevel == FeatureLevel.Level_11_0)
                {
                    CompileLevelPS="ps_5_0";
                    CompileLevelVS = "vs_5_0";
                }
                else
                {
                    CompileLevelPS = "ps_4_0";
                    CompileLevelVS = "vs_4_0";
                }

                try
                {
                    /// compile the shader to byte code
                    m_PixelShaderByteCode = ShaderBytecode.CompileFromFile(
                                                ShaderRootPath + Name,       /// File Path of the file containing the code 
                                                PS_EntryPoint,               /// The entry point for the shader
                                                CompileLevelPS,              /// What specifications (shader version) to compile with
                                                sf, EffectFlags.None, null, includeFX);

                    /// compile the shader to byte code
                    m_VertexShaderByteCode = ShaderBytecode.CompileFromFile(
                                                ShaderRootPath + Name,       /// File Path of the file containing the code 
                                                VS_EntryPoint,               /// The entry point for the shader
                                                CompileLevelVS,              /// What specifications (shader version) to compile with
                                                sf, EffectFlags.None, null, includeFX);
                }
                catch (Exception ex)
                {
                    System.Windows.Forms.MessageBox.Show(ex.Message);

                }
            }

            /// init effect 
            m_effect = new FXEffect( dev, m_PixelShaderByteCode, m_VertexShaderByteCode );

            /// init all the variables 
            InitVariables();
        }

        private void InitVariables()
        {
            m_ViewConstantBuffer = m_effect.GetConstantBufferByName( "cbViewMatrix" );


            /// Obtain the variables from effect
            m_WorldInverseTranspose = m_ViewConstantBuffer.GetMemberByName<Matrix>( "g_mWorldInverseTrans" );
            m_ViewInverse = m_ViewConstantBuffer.GetMemberByName<Matrix>( "g_mViewInverse" );

            /// Matrix transformations
            m_WorldMatrixVariable = m_ViewConstantBuffer.GetMemberByName<Matrix>( "g_mWorld" );
            m_ViewMatrixVariable = m_ViewConstantBuffer.GetMemberByName<Matrix>( "g_mView" );
            m_WorldViewMatrixVariable = m_ViewConstantBuffer.GetMemberByName<Matrix>( "g_mWorldView" );
            m_WorldViewProjectionMatrixVariable = m_ViewConstantBuffer.GetMemberByName<Matrix>( "g_mWorldViewProjection" );

            /// Camera Position
            m_CameraPosition = m_ViewConstantBuffer.GetMemberByName<Vector3>("g_vCameraPosition");

            /// Shader Variables
            m_MaterialBuffer = m_effect.GetConstantBufferByName( "cbMaterial" );
            m_Diffuse = m_MaterialBuffer.GetMemberByName<Vector3>( "g_vMaterialDiffuse" );
            m_Ambient = m_MaterialBuffer.GetMemberByName<Vector3>( "g_vMaterialAmbient" );
            m_Opacity = m_MaterialBuffer.GetMemberByName<float>( "g_fMaterialAlpha" );
            m_Specular = m_MaterialBuffer.GetMemberByName<Vector3>( "g_vMaterialSpecular" );
            m_SpecularPower = m_MaterialBuffer.GetMemberByName<float>( "g_nMaterialShininess" );
        }


        /// <summary>
        /// Compile in runtime the shader that 
        /// get from the memory
        /// </summary>
        /// <param name="Shader"></param>
        /// <returns>The errors from the building</returns>
        public virtual String RunTimeCompile(String Shader){

            ShaderBytecode bytecode = null;

            try {
                IncludeFX includeFX = new IncludeFX();
                /// compile the shader to byte code
                bytecode = ShaderBytecode.Compile(
                                            Shader,     /// string buffer that containing the code 
                                            "fx_5_0",   /// What specifications (shader version) to compile with
                                            ShaderFlags.None, EffectFlags.None, null, includeFX);


            } catch (Exception ex) {
                return ex.Message;
            }

            /// check if the shader compile correct
            if (bytecode != null) {
                /// create the effect variable
                //m_effect = new Effect(Engine.g_device, bytecode);

                /// init all the variables 
                InitVariables();
            }

            return "";
        }

        /// <summary>
        /// set the position of the mesh base on the world matrix
        /// all the change pass to the shader
        /// </summary>
        /// <param name="m_WorldMatrix"></param>
        public void SetThePositions(Matrix m_WorldMatrix)
        {
     		// set the world matrix
            m_WorldMatrixVariable.Set(m_WorldMatrix);

            Matrix viewMatrix = Engine.g_RenderCamera.ViewMatrix;

            /// set the world*view*Projection matrix
            m_WorldViewProjectionMatrixVariable.Set(m_WorldMatrix * viewMatrix * Engine.g_RenderCamera.ProjMatrix);

            /// set the world*view matrix
            m_WorldViewMatrixVariable.Set( m_WorldMatrix * viewMatrix);

            /// set the view matrix
            m_ViewMatrixVariable.Set(viewMatrix);

            viewMatrix.Invert();

            m_ViewInverse.Set(viewMatrix);

            /// set the position of the camera
            m_CameraPosition.Set(Engine.g_RenderCamera.Eye);

            /// set the world inverse transpose
            m_WorldInverseTranspose.Set(Matrix.Transpose(Matrix.Invert(m_WorldMatrix)));

        }

        /// <summary>
        /// Execute all the passes of the shader
        /// </summary>
        public void Execute( DeviceContext deviceContext )
        {
            m_effect.Apply( deviceContext );
        }
        

        /// <summary>
        /// Execute a specific passe of the shader
        /// </summary>
        /// <param name="index"></param>
        public void Execute(int index)
        {
            /// execute the pass
            //m_technique.GetPassByIndex(index).Apply(Engine.g_device.ImmediateContext);
        }

        /// <summary>
        /// Free resources
        /// </summary>
        public void Dispose()
        {
            m_effect.Dispose();
        }

    }
}
