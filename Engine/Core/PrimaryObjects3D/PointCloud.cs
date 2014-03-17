using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

/// SharpDX includes
using SharpDX;
using SharpDX.Direct3D11;
using SharpDX.DXGI;

/// resolve conflicts
using Device = SharpDX.Direct3D11.Device;
using Buffer = SharpDX.Direct3D11.Buffer;

using System.Runtime.InteropServices;
using SharpDX.D3DCompiler;
using SharpDX.Direct3D;
using GraphicsEngine.Core;

namespace FxGraphicsEngine.Core.PrimaryObjects3D
{
    public class PointCloud : Object3D
    {
        #region Variables

        #region Public

        /// <summary>
        /// The shader that use this mesh
        /// </summary>
        public Shader m_shader;

        #endregion

        #region Private

        /// <summary>
        /// Name of the mesh
        /// </summary>
        protected String _name = "";

        /// <summary>
        /// Position of the mesh
        /// </summary>
        protected Vector3 _position;

        /// <summary>
        /// Rotation of the mesh
        /// </summary>
        protected Vector3 _rotation;

        /// <summary>
        /// Scaling of the mesh
        /// </summary>
        private Vector3 _scale;

        /// <summary>
        /// Define the input layout
        /// </summary>
        private static InputElement[] m_layout =
        {
				new InputElement("POSITION",0,Format.R32G32B32_Float,0,0),
				new InputElement("NORMAL",0,Format.R32G32B32_Float,12,0),
                new InputElement("TEXCOORD",0,Format.R32G32_Float,24,0),
                new InputElement("TANGENT",0,Format.R32G32B32_Float,32,0),
                new InputElement("BINORMAL",0,Format.R32G32B32_Float,44,0),
        };
        #endregion

        #endregion

        public override void Render(SharpDX.Direct3D11.DeviceContext devCont)
        {
            throw new NotImplementedException();
        }

        public override void Dispose()
        {
            throw new NotImplementedException();
        }

        public override void Update()
        {
            throw new NotImplementedException();
        }






    }
}
