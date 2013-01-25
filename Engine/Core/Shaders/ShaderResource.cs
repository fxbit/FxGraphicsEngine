using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using SlimDX.D3DCompiler;
using SlimDX.Direct3D11;

namespace GraphicsEngine.Core.Shaders
{
    public class ShaderResource
    {
        public readonly int Slot;
        public readonly Boolean isPsExist;
        public readonly Boolean isVsExist;
        private ShaderResourceView localSRV;
        private Boolean isDirty = false;

        public ShaderResource( String resource_name, ShaderReflection psReflection, ShaderReflection vsReflection )
        {
            // try to find it in pixel shader
            try {

                // find the bind point of the resource
                Slot = psReflection.GetResourceBindingDescription( resource_name ).BindPoint;

                // set that the resource exist
                isPsExist = true;

            } catch ( Exception ex ) {

                // set the resource unusable
                isPsExist = false;
            }
            
            // try to find it in vertex shader
            try {

                // find the bind point of the resource
                Slot = vsReflection.GetResourceBindingDescription( resource_name ).BindPoint;

                // set that the resource exist
                isVsExist = true;

            } catch ( Exception ex ) {

                // set the resource unusable
                isVsExist = false;
            }
        }

        public void SetResource( ShaderResourceView srv )
        {
            // store the srv local
            localSRV = srv;
        }

        public void Commit( DeviceContext deviceContext )
        {
            // if is exist in ps the commit the resource
            if ( isPsExist ) {
                deviceContext.PixelShader.SetShaderResource( localSRV, Slot );
            }
            // if is exist in vs the commit the resource
            if ( isVsExist ) {
                deviceContext.VertexShader.SetShaderResource( localSRV, Slot );
            }
        }
    }
}
