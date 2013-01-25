using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using SlimDX;
using SlimDX.DXGI;
using SlimDX.Direct3D11;

// resolve conflict - DXGI.Device & Direct3D10.Device
using Device = SlimDX.Direct3D11.Device;
using Buffer = SlimDX.Direct3D11.Buffer;
using Effect = SlimDX.Direct3D11.Effect;
using EffectFlags = SlimDX.D3DCompiler.EffectFlags;
using GraphicsEngine.Managers;
using System.Drawing;
using FXFramework;

namespace GraphicsEngine.Core.Shaders.Modules {
    public class ShaderVariable_Brick : ShaderVariable_Base {

        private float _BrickWidth = 0.2f;
        private float _BrickHeight = 0.1f;
        private float _BrickShift = 0.5f;
        private float _MortarThickness = 0.1f;

        #region Effect variables
        private FXVariable<float> BrickWidth_Variable;
        private FXVariable<float> BrickHeight_Variable;
        private FXVariable<float> BrickShift_Variable;
        private FXVariable<float> MortarThickness_Variable;
        #endregion

        #region Public properties
        public float BrickWidth
        {
            set { _BrickWidth = value; BrickWidth_Variable.Set(_BrickWidth); }
            get { return _BrickWidth; }
        }

        public float BrickHeight
        {
            set { _BrickHeight = value; BrickHeight_Variable.Set(_BrickHeight); }

            get { return _BrickHeight; }
        }

        public float BrickShift
        {
            set { _BrickShift = value; BrickShift_Variable.Set(_BrickShift); }

            get { return _BrickShift; }
        }

        public float MortarThickness
        {
            set { _MortarThickness = value; MortarThickness_Variable.Set(_MortarThickness); }
            get { return _MortarThickness; }
        }
        #endregion


        public ShaderVariable_Brick(String VariableName)
        {
            this.VariableName = VariableName;
        }

        public override void Init( FXEffect m_effect, FXConstantBuffer m_cb )
        {
            // bind the local variables with the shader
            BrickWidth_Variable = m_cb.GetMemberByName<float>( VariableName + "_BrickWidth" );
            BrickHeight_Variable = m_cb.GetMemberByName<float>( VariableName + "_BrickHeight" );
            BrickShift_Variable = m_cb.GetMemberByName<float>( VariableName + "_BrickShift" );
            MortarThickness_Variable = m_cb.GetMemberByName<float>( VariableName + "_MortarThickness" );
            
            // set the default values
            BrickWidth_Variable.Set(_BrickWidth);
            BrickHeight_Variable.Set(_BrickHeight);
            BrickShift_Variable.Set(_BrickShift);
            MortarThickness_Variable.Set(_MortarThickness);

        }

    }
}
