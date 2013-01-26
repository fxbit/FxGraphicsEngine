using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

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

    [StructLayout( LayoutKind.Explicit, Size = 32 )] //20
    public struct cbPerlinNoise
    {
        [FieldOffset( 0 )]
        public float MinValue;

        [FieldOffset( 4 )]
        public float MaxValue;

        [FieldOffset( 8 )]
        public float Tile_X;

        [FieldOffset( 12 )]
        public float Tile_Y;

        [FieldOffset( 16 )]
        public int Loops;
    }

    public class ShaderPerlin : Shader {

        /// <summary>
        /// the name of the noise that we want
        /// </summary>
        const String ShaderName = "FX/perlinNoise.fx";

        #region Private effect variable

        // variable for the perlin noise
        private FXConstantBuffer<cbPerlinNoise> CB_PerlinNoise;
        private cbPerlinNoise localPerlinNoise_Variable;
        private FXResourceVariable RandomTex_Variable;

        #endregion

        #region private variables
        private float _MinValue;
        private float _MaxValue;
        private float _Tile_X;
        private float _Tile_Y;
        private int _Loops;
        private int _Seed;
        #endregion

        #region Public variables
        public float MinValue
        {
            set { _MinValue = value; localPerlinNoise_Variable.MinValue = _MinValue; CB_PerlinNoise.UpdateValue( localPerlinNoise_Variable ); }
            get { return _MinValue; }
        }

        public float MaxValue
        {
            set { _MaxValue = value; localPerlinNoise_Variable.MaxValue = _MaxValue; CB_PerlinNoise.UpdateValue( localPerlinNoise_Variable ); }
            get { return _MaxValue; }
        }

        public float Tile_X
        {
            set { _Tile_X = value; localPerlinNoise_Variable.Tile_X = _Tile_X; CB_PerlinNoise.UpdateValue( localPerlinNoise_Variable ); }
            get { return _Tile_X; }
        }

        public float Tile_Y
        {
            set { _Tile_Y = value; localPerlinNoise_Variable.Tile_Y = _Tile_Y; CB_PerlinNoise.UpdateValue( localPerlinNoise_Variable ); }
            get { return _Tile_Y; }
        }

        public int Loops
        {
            set { _Loops = value; localPerlinNoise_Variable.Loops = _Loops; CB_PerlinNoise.UpdateValue( localPerlinNoise_Variable ); }
            get { return _Loops; }
        }

        public int Seed
        {
            set { _Seed = value; SetRandomTex( value ); }
            get { return _Seed; }
        }
        #endregion

        #region Constructor
        public ShaderPerlin()
            : base(ShaderName)
        {
            _Seed = 10;
            InitShader();
        }

        public ShaderPerlin(String Name)
            : base(Name)
        {
            InitShader();
        }
        #endregion

        #region Init Shader

        private void InitShader()
        {
            /// Bind the random Texture
            RandomTex_Variable =m_effect.GetResourceByName( "randomTex" );

            /// Bind the constant buffer with local buffer
            CB_PerlinNoise = m_effect.GetConstantBufferByName<cbPerlinNoise>( "cbPerlinNoiseVariables");

            // create and set random texture
            SetRandomTex( _Seed );
        }

        #endregion

        #region Private functions

        private void SetRandomTex(int seed)
        {
            // set the random size
            int RandNum_Size = 256;

            // set the description of the texture
            Texture1DDescription RandomTex_Desc = new Texture1DDescription();
            RandomTex_Desc.Format = Format.R8_UInt;
            RandomTex_Desc.CpuAccessFlags = CpuAccessFlags.None;
            RandomTex_Desc.Width = RandNum_Size * 2;
            RandomTex_Desc.Usage = ResourceUsage.Default;
            RandomTex_Desc.OptionFlags = ResourceOptionFlags.None;
            RandomTex_Desc.MipLevels = 1;
            RandomTex_Desc.BindFlags = BindFlags.ShaderResource;
            RandomTex_Desc.ArraySize = 1;

            // start the stream 
            DataStream stream = new DataStream(RandNum_Size * 2, true, true);

            // Initialize the random generator
            Random generator = new Random( seed );

            // allocate the random values 
            byte[] RandNum_Host = new byte[RandNum_Size * 2];

            // Copy the source data twice to the generator array
            for (int i = 0; i < RandNum_Size; i++) {
                RandNum_Host[i] = (byte)generator.Next(256);
                RandNum_Host[i + RandNum_Size] = RandNum_Host[i];
            }

            // pass the randoms to the stream
            stream.WriteRange<byte>(RandNum_Host);
            stream.Position = 0;

            // create the texture and pass the data
            Texture1D RandomTex = new Texture1D(Engine.g_device, RandomTex_Desc, stream);

            // close the stream we don't need it any more
            stream.Close();

            // create the resource view to be able to pass it to the shader
            ShaderResourceView RandomResourceView = new ShaderResourceView(Engine.g_device, RandomTex);

            // set the Resource to the shader
            RandomTex_Variable.SetResource(RandomResourceView);
        }
        
        #endregion

    }
}
