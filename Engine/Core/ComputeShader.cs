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
using SharpDX.D3DCompiler;
using System.IO;
using FxMaths;
using FXFramework;
using SharpDX.Direct3D;

namespace GraphicsEngine.Core {

    public class ComputeShader {

        String ShaderRootPath = System.Windows.Forms.Application.StartupPath + "\\";

        #region Shader's Variables


        /// <summary>
        /// Shader Effect
        /// </summary>
        public FXEffect m_effect;


        #endregion



        #region CS struct Sizes
        /// <summary>
        /// Size in bytes of CS type Float1 
        /// </summary>
        public static int SizeOfFloat1
        {
            get { return 4; }
        }
        /// <summary>
        /// Size in bytes of CS type Float2
        /// </summary>
        public static int SizeOfFloat2
        {
            get { return 8; }
        }
        /// <summary>
        /// Size in bytes of CS type Float4
        /// </summary>
        public static int SizeOfFloat3
        {
            get { return 12; }
        }
        /// <summary>
        /// Size in bytes of CS type Float4
        /// </summary>
        public static int SizeOfFloat4
        {
            get { return 16; }
        }
        /// <summary>
        /// Size in bytes of CS type Int1
        /// </summary>
        public static int SizeOfInt1
        {
            get { return 4; }
        }
        /// <summary>
        /// Size in bytes of CS type Int2
        /// </summary>
        public static int SizeOfInt2
        {
            get { return 8; }
        }
        /// <summary>
        /// Size in bytes of CS type Int4
        /// </summary>
        public static int SizeOfInt4
        {
            get { return 16; }
        }
        /// <summary>
        /// Size in bytes of CS type Char4
        /// </summary>
        public static int SizeOfChar4
        {
            get { return 4; }
        }
        #endregion



        #region Constructions 

        /// <summary>
        /// Compile Runtime the shader
        /// </summary>
        /// <param name="Name"></param>
        /// <param name="entryPoint"></param>
        public ComputeShader( String Name, String entryPoint , String IncludePath =  null)
        {

            Device dev = Engine.g_device;

            // init the include class and subpath
            IncludeFX includeFX = new IncludeFX();
            if ( IncludePath == null ) {
                includeFX.IncludeDirectory = System.Windows.Forms.Application.StartupPath + "\\ComputeHLSL\\";
            } else {
                includeFX.IncludeDirectory = IncludePath;
            }
            ShaderBytecode bytecode= null;

            ShaderFlags sf;

            // select the shaders flags for 
            if ( Settings.Debug ) {
                sf = ShaderFlags.SkipOptimization | ShaderFlags.Debug | ShaderFlags.PreferFlowControl;
            } else {
                sf = ShaderFlags.OptimizationLevel3;
            }

            // set the compile level base on running Feature level.
            String CompileLevelCS;
            if (Settings.FeatureLevel == FeatureLevel.Level_11_0)
            {
                CompileLevelCS = "cs_5_0";
            }
            else
            {
                CompileLevelCS = "cs_4_0";
            }


            /// compile the shader to byte code
            bytecode = ShaderBytecode.CompileFromFile(
                                        ShaderRootPath + Name,               /// File Path of the file containing the code 
                                        entryPoint,                          /// The name of the executable function
                                        CompileLevelCS,                      /// What specifications (shader version) to compile with cs_4_0 for directX10 and cs_5_0 for directx11
                                        sf, EffectFlags.None, null, includeFX);


            // init effect 
            m_effect = new FXEffect( dev, csByteCode: bytecode );

        }

        /// <summary>
        /// Precompile the shader
        /// </summary>
        /// <param name="Name"></param>
        public ComputeShader( String Name )
        {

            Device dev = Engine.g_device;

            IncludeFX includeFX = new IncludeFX();
            ShaderBytecode bytecode= null;

            // check that the shader is exist
            if ( File.Exists( ShaderRootPath + Name ) ) {
                // open the file to read it 
                FileStream fileStream = new FileStream( ShaderRootPath + Name, FileMode.Open );

                // allocate the byte stream
                byte []fileByte = new byte[fileStream.Length];

                // read the file stream
                fileStream.Read( fileByte, 0, (int)fileStream.Length );

                // close the file stream
                fileStream.Close();

                bytecode = new ShaderBytecode(fileByte);
                
            } else {
                System.Windows.Forms.MessageBox.Show( "Shader:" + ShaderRootPath + Name + "   is not exist " );

                return;
            }


            // init effect 
            m_effect = new FXEffect( dev, csByteCode: bytecode );
        }

        #endregion 



        #region Execute

        /// <summary>
        /// Execute the shader
        /// </summary>
        public void Execute(int threadsX,int threadsY)
        {
            Device dev = Engine.g_device;

            // set the shader variables
            m_effect.Apply( dev.ImmediateContext );

            // execute the shader in groups
            dev.ImmediateContext.Dispatch(threadsX, threadsY, 1);

            // clean the binging of srv/uav 
            m_effect.CleanBind( dev.ImmediateContext );
        }

        #endregion



        #region Dispose

        /// <summary>
        /// Free resources
        /// </summary>
        public void Dispose()
        {
            m_effect.Dispose();
        }

        #endregion 



        #region Static Create Buffers

        /// <summary>
        /// Create buffer base on data stream and specific struct
        /// </summary>
        /// <param name="data"></param>
        /// <param name="type"></param>
        public static Buffer CreateBuffer(int numElements, int sizeOfElementInByte, AccessViewType type, DataStream data = null)
        {
            // set the buffer desc
            BufferDescription BuffDescription = new BufferDescription
            {
                BindFlags = BindFlags.None,
                StructureByteStride = sizeOfElementInByte,
                SizeInBytes = numElements * sizeOfElementInByte,
                OptionFlags = ResourceOptionFlags.BufferStructured,
            };

            if (type.HasFlag(AccessViewType.SRV))
                BuffDescription.BindFlags |= BindFlags.ShaderResource;

            if (type.HasFlag(AccessViewType.UAV))
                BuffDescription.BindFlags |= BindFlags.UnorderedAccess;

            if (type.HasFlag(AccessViewType.UAV))
                BuffDescription.Usage = ResourceUsage.Default;

            // create the buffer
            Buffer Output;
            if (data == null)
            {
                Output = new Buffer(Engine.g_device, BuffDescription);
            }
            else
            {
                // reset the seek point
                data.Seek(0, System.IO.SeekOrigin.Begin);

                Output = new Buffer(Engine.g_device, data, BuffDescription);
            }

            return Output;
        }

        /// <summary>
        /// Create a staging buffer base on regular buffer. That buffer must be UAV
        /// </summary>
        /// <param name="gpuBuffer"></param>
        /// <returns></returns>
        public static Buffer CreateStagingBuffer( Buffer gpuBuffer )
        {
            // create the buffer description
            BufferDescription stagingBufferDescription = new BufferDescription
            {
                BindFlags = BindFlags.None,
                OptionFlags = ResourceOptionFlags.BufferStructured,
                CpuAccessFlags = CpuAccessFlags.Read | CpuAccessFlags.Write,
                Usage = ResourceUsage.Staging,
                SizeInBytes = gpuBuffer.Description.SizeInBytes,
                StructureByteStride = gpuBuffer.Description.StructureByteStride,
            };

            // create staging Buffer
            Buffer stagingBuffer = new Buffer( Engine.g_device, stagingBufferDescription );

            return stagingBuffer;
        }


        #endregion

    }
}
