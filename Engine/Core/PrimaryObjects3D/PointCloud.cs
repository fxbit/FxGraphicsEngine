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
using FXFramework;

namespace GraphicsEngine.Core.PrimaryObjects3D
{
    [StructLayout(LayoutKind.Explicit, Size = 24)]
    public struct PointCloudParticle
    {
        /// <summary>
        /// Position of the Particle
        /// </summary>
        [FieldOffset(0)]
        public Vector3 Pos;

        /// <summary>
        /// Color of the Particle
        /// </summary>
        [FieldOffset(12)]
        public Vector3 Color;
    }

    public class PointCloud : Object3D
    {
        #region Variables

        #region Public

        #endregion

        #region Private

        /// <summary>
        /// Name of the mesh
        /// </summary>
        protected String _name = "";

        /// <summary>
        /// Position of the mesh
        /// </summary>
        protected Vector3 _position = new Vector3();

        /// <summary>
        /// Rotation of the mesh
        /// </summary>
        protected Vector3 _rotation = new Vector3();

        /// <summary>
        /// Scaling of the mesh
        /// </summary>
        private Vector3 _scale = new Vector3(1.0f);

        /// <summary>
        /// The shader that use this mesh
        /// </summary>
        private Shader m_shader;

        /// <summary>
        /// Buffer with the Particles
        /// </summary>
        protected Buffer m_BufferParticles = null;

        /// <summary>
        /// Access view for our particles
        /// </summary>
        protected ShaderResourceView m_srvBufferParticles;

        /// <summary>
        /// FxResource variable for buffer particles
        /// </summary>
        protected FXResourceVariable m_rvBufferParticles;

        /// <summary>
        /// The number of the particles
        /// </summary>
        protected int m_numParticles;
        #endregion

        #endregion

        public PointCloud(List<FxMaths.Vector.FxVector3f> Points, List<FxMaths.Vector.FxVector3f> Colors)
        {
            Device dev = Engine.g_device;

            /// make a stream with the vertex to be able to insert it to the mesh
            DataStream stream = new DataStream(Marshal.SizeOf(typeof(PointCloudParticle)) * Points.Count, true, true);

            /// Init the shader
            m_shader = new Shaders.ShaderPointCloud();
            m_numParticles = Points.Count;

            /// write the particles to the stream
            for (int i = 0; i < m_numParticles; i++)
            {
                Points[i].WriteToDataStream(stream);
                Colors[i].WriteToDataStream(stream);
            }
            /// reset the position in the stream
            stream.Position = 0;

            /// Fill the buffer with the vertices
            m_BufferParticles = ComputeShader.CreateBuffer(m_numParticles, ComputeShader.SizeOfFloat3 * 2, AccessViewType.SRV, stream);
            m_BufferParticles.DebugName = "ParticleBuffer";

            m_srvBufferParticles = FXResourceVariable.InitSRVResource(dev, m_BufferParticles);
            m_rvBufferParticles = m_shader.m_effect.GetResourceByName("particleBuffer");
            m_rvBufferParticles.SetResource(m_srvBufferParticles);

            // set the world matrix
            m_WorldMatrix = Matrix.Scaling(_scale) * Matrix.RotationYawPitchRoll(_rotation.Y, _rotation.X, _rotation.Z) * Matrix.Translation(_position);

            // close the stream
            stream.Close();
        }


        public override void Render(SharpDX.Direct3D11.DeviceContext devCont)
        {
            Device dev = Engine.g_device;

            /// Clear the layout to the engine
            devCont.InputAssembler.InputLayout = null;

            /// set the type of primitive(triangleList)
            devCont.InputAssembler.PrimitiveTopology = PrimitiveTopology.PointList;

            /// set the position of the mesh base on the world matrix
            /// all the change pass to the shader
            m_shader.SetThePositions(m_WorldMatrix);

            /// Execute all the passes of the shader
            m_shader.Execute(devCont);

            /// Render
            devCont.Draw(m_numParticles, 0);
        }

        public override void Dispose()
        {
            m_shader.Dispose();
        }

        public override void Update() { }
    }
}
