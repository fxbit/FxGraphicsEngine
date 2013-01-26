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

    public class ShaderVariable_Perlin : ShaderVariable_Base {

        #region  variable for the perlin noise
        private FXVariable<float> MinValue_Variable;
        private FXVariable<float> MaxValue_Variable;
        private FXVariable<float> Tile_X_Variable;
        private FXVariable<float> Tile_Y_Variable;
        private FXVariable<int> Loops_Variable;

        private FXResourceVariable RandomTex_Variable;
        #endregion

        #region private variables
        private float _MinValue=0f;
        private float _MaxValue=1f;
        private float _Tile_X=1;
        private float _Tile_Y=1;
        private int _Loops=5;
        private int _Seed = 10;
        #endregion

        #region Public variables
        public float MinValue
        {
            set { _MinValue = value; MinValue_Variable.Set(_MinValue); }
            get { return _MinValue; }
        }

        public float MaxValue
        {
            set { _MaxValue = value; MaxValue_Variable.Set(_MaxValue); }
            get { return _MaxValue; }
        }

        public float Tile_X
        {
            set { _Tile_X = value; Tile_X_Variable.Set(_Tile_X); }
            get { return _Tile_X; }
        }

        public float Tile_Y
        {
            set { _Tile_Y = value; Tile_Y_Variable.Set(_Tile_Y); }
            get { return _Tile_Y; }
        }

        public int Loops
        {
            set { _Loops = value; Loops_Variable.Set(_Loops); }
            get { return _Loops; }
        }

        public int Seed
        {
            set { _Seed = value; SetRandomTex(_Seed); }
            get { return _Seed; }
        }
        #endregion

        public ShaderVariable_Perlin(String VariableName)
        {
            this.VariableName = VariableName;
        }

        public override void Init( FXEffect m_effect , FXConstantBuffer m_cb )
        {
            /// Bind the random Texture
            RandomTex_Variable = m_effect.GetResourceByName(VariableName + "_RandomTex");

            /// Get the variables from Constant buffer
            MinValue_Variable = m_cb.GetMemberByName<float>( VariableName + "_MinValue" );
            MaxValue_Variable = m_cb.GetMemberByName<float>( VariableName + "_MaxValue" );
            Tile_X_Variable = m_cb.GetMemberByName<float>( VariableName + "_Tile_X" );
            Tile_Y_Variable = m_cb.GetMemberByName<float>( VariableName + "_Tile_Y" );
            Loops_Variable = m_cb.GetMemberByName<int>( VariableName + "_Loops" );

            // create and set random texture
            SetRandomTex(_Seed);

            // set the values of the variables
            MinValue_Variable.Set(_MinValue);
            MaxValue_Variable.Set(_MaxValue);
            Loops_Variable.Set(_Loops);
            Tile_X_Variable.Set(_Tile_X);
            Tile_Y_Variable.Set(_Tile_Y);
        }

        private void SetRandomTex(int Seed)
        {
            // set the seed
            _Seed = Seed;

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
            Random generator = new Random(Seed);

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
    }

}
