using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

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

namespace GraphicsEngine.Core.PrimaryObjects3D {
    public class Mesh : Object3D {

        #region Variables

        #region Public
        /// <summary>
        /// List with the vertices
        /// </summary>
        public List<Polygon> m_polygons;

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
        /// The input layout the the Shader understand
        /// </summary>
        protected InputLayout m_input_layout;

        /// <summary>
        /// Buffer with the Vertices of our mesh
        /// </summary>
        protected Buffer m_BufferVertices = null;

        /// <summary>
        /// Class that bind the vertex buffer with the drawing
        /// </summary>
        protected VertexBufferBinding m_VertexBufferBinding;

        /// <summary>
        /// Buffer with the Indices of our mesh
        /// </summary>
        protected Buffer m_BufferIndices = null;


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

        #region Functions

        public override void Update() { }

        #region Constractor
        /// <summary>
        /// Constructor, instantiates empty lists
        /// </summary>
        public Mesh()
        {
            m_polygons = new List<Polygon>();
            m_BoundaryBox = new BoundingBox();
            _scale = new Vector3(1);
        }
        #endregion

        /// <summary>
        /// Free resources
        /// </summary>
        public override void Dispose()
        {
            /// the internal mesh
            m_polygons.Clear();
            m_shader.Dispose();
            m_input_layout.Dispose();
        }

        #region Add vertices
        public void AddPolygon(Polygon ver, Boolean hasTangent)
        {
            /// find the boundary box
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    if (m_BoundaryBox.Maximum[j] < ver[i].position[j])
                        m_BoundaryBox.Maximum[j] = ver[i].position[j];

                    if (m_BoundaryBox.Minimum[j] > ver[i].position[j])
                        m_BoundaryBox.Minimum[j] = ver[i].position[j];
                }
            }

            /// check if we have tangent
            if (!hasTangent) {
                #region old way to get tangent
                // we must calculate the tangent and the binormal
                // base to the : http://www.terathon.com/code/tangent.html
                /*
                Vector3 v1 = ver.b.position - ver.a.position;
                Vector3 v2 = ver.c.position - ver.a.position;

                Vector2 tex1 = ver.b.textCoords - ver.a.textCoords;
                Vector2 tex2 = ver.c.textCoords - ver.a.textCoords;

                float r = 1.0F / (tex1.X * tex2.Y - tex2.X * tex1.Y);

                Vector3 sdir=new Vector3((tex2.Y * v1.X - tex1.Y * v2.X) * r, (tex2.Y * v1.Y - tex1.Y * v2.Y) * r,
                (tex2.Y * v1.Z - tex1.Y * v2.Z) * r);

                Vector3  tdir=new Vector3((tex1.X * v2.X - tex2.X * v1.X) * r, (tex1.X * v2.Y - tex2.X * v1.Y) * r,
                (tex1.X * v2.Z - tex2.X * v1.Z) * r);

                // Gram-Schmidt orthogonalize
                ver.a.tangent = sdir - ver.a.normal * Vector3.Dot(ver.a.normal, sdir);
                ver.a.tangent.Normalize();
                ver.b.tangent = sdir - ver.b.normal * Vector3.Dot(ver.b.normal, sdir);
                ver.b.tangent.Normalize();
                ver.c.tangent = sdir - ver.c.normal * Vector3.Dot(ver.c.normal, sdir);
                ver.c.tangent.Normalize();

                //calc the binormal
                float w = (Vector3.Dot(Vector3.Cross(ver.a.normal, sdir), tdir) < 0.0f) ? -1.0F : 1.0F;
                ver.a.binormal = Vector3.Cross(ver.a.normal, ver.a.tangent) * w;
                ver.a.binormal.Normalize();

                w = (Vector3.Dot(Vector3.Cross(ver.b.normal, sdir), tdir) < 0.0f) ? -1.0F : 1.0F;
                ver.b.binormal = Vector3.Cross(ver.b.normal, ver.b.tangent) * w;
                ver.b.binormal.Normalize();

                w = (Vector3.Dot(Vector3.Cross(ver.c.normal, sdir), tdir) < 0.0f) ? -1.0F : 1.0F;
                ver.c.binormal = Vector3.Cross(ver.c.normal, ver.c.tangent) * w;
                ver.c.binormal.Normalize();
                */
                #endregion

                //////////////////////////////////////////////////////////
                /// http://jerome.jouvie.free.fr/OpenGl/Lessons/Lesson8.ph
                Vector3[] tangent = new Vector3[3];
                Vector3[] binormal = new Vector3[3];
                Vector3[] normal = new Vector3[3];
                for (int i = 0; i < 3; i++) {
                    Vector3 p21 = ver[1 + i].position - ver[i].position;
                    Vector3 p31 = ver[2 + i].position - ver[i].position;

                    Vector2 uv21 = ver[1 + i].textCoords - ver[i].textCoords;
                    Vector2 uv31 = ver[2 + i].textCoords - ver[i].textCoords;

                    tangent[i] = Vector3.Subtract(Vector3.Multiply(p21, uv31.Y), Vector3.Multiply(p31, uv21.Y));
                    tangent[i].Normalize();
                    binormal[i] = Vector3.Subtract(Vector3.Multiply(p31, uv21.X), Vector3.Multiply(p21, uv31.X));
                    binormal[i].Normalize();
                    normal[i] = Vector3.Cross(tangent[i], binormal[i]);

                    /// Gram-Schmidt orthogonalization
                    tangent[i] -= Vector3.Multiply(normal[i], Vector3.Dot(normal[i], tangent[i]));
                    tangent[i].Normalize();

                }
                ver.a.tangent = tangent[0];
                ver.b.tangent = tangent[1];
                ver.c.tangent = tangent[2];

                ver.a.binormal = binormal[0];
                ver.b.binormal = binormal[1];
                ver.c.binormal = binormal[2];

                ver.a.normal = normal[0];
                ver.b.normal = normal[1];
                ver.c.normal = normal[2];
            }

            m_polygons.Add(ver);
        }

        public void AddPolygon(Vertex a, Vertex b, Vertex c)
        {
            m_polygons.Add(new Polygon(a, b, c));
        }

        public void AddPolygon(Vertex a, Vertex b, Vertex c, bool hasTangent)
        {
            Polygon poly = new Polygon(a, b, c);
            AddPolygon(poly, hasTangent);
        }
        #endregion

        #region Position Rotation Scale
        public void SetPosition(Vector3 vec)
        {
            _position = vec;
            UpdateWorld();
        }

        public void SetRotation(Vector3 value)
        {
            _rotation = value;
            UpdateWorld();
        }

        public void SetScale(Vector3 value)
        {
            _scale = value;
            UpdateWorld();
        }

        /// <summary>
        /// Update the world matrix (the position of the mesh to the world)
        /// </summary>
        private void UpdateWorld()
        {
            m_WorldMatrix = Matrix.Scaling(_scale) * Matrix.RotationYawPitchRoll(_rotation.Y, _rotation.X, _rotation.Z) * Matrix.Translation(_position);
        }
        #endregion

        /// <summary>
        /// Create the mesh and stream it to the device
        /// </summary>
        public void CreateMesh()
        {
            Device dev = Engine.g_device;

            /// make a stream with the vertex to be able to insert it to the mesh
            DataStream stream = new DataStream(Marshal.SizeOf(typeof(Polygon)) * m_polygons.Count * 3, true, true);

            /// write the polygons to the stream
            stream.WriteRange<Polygon>(m_polygons.ToArray());

            /// reset the position in the stream
            stream.Position = 0;

            /// Fill the buffer with the vertices
            m_BufferVertices = new SharpDX.Direct3D11.Buffer(dev, stream, new BufferDescription() {
                BindFlags = BindFlags.VertexBuffer,
                CpuAccessFlags = CpuAccessFlags.None,
                OptionFlags = ResourceOptionFlags.None,
                SizeInBytes = (int)stream.Length,
                Usage = ResourceUsage.Default
            });

            // create the binder for the vertex 
            m_VertexBufferBinding = new VertexBufferBinding(m_BufferVertices, 56 /* the size of the Vertex */ , 0);

            // close the stream
            stream.Close();

            /*
            /// make a stream with the Indices to be able to insert it to the mesh
            stream = new DataStream(sizeof(uint) * m_polygons.Count * 3, true, true);
            for (uint i = 0; i < m_polygons.Count * 3; i++)
                stream.Write<uint>(i);

            /// reset the position in the stream
            stream.Position = 0;

            m_BufferIndices = new SharpDX.Direct3D11.Buffer(dev, stream, new BufferDescription() {
                BindFlags = BindFlags.IndexBuffer,
                CpuAccessFlags = CpuAccessFlags.None,
                OptionFlags = ResourceOptionFlags.None,
                SizeInBytes = (int)stream.Length,
                Usage = ResourceUsage.Default
            });

            // close the stream
            stream.Close();
            */

            // create the input layout
            m_input_layout = new InputLayout(
                Engine.g_device,
                m_shader.m_VertexShaderByteCode,
                m_layout);
        }

        

        /// <summary>
        /// Draw the 3d mesh to screen
        /// </summary>
        public override void Render( DeviceContext devCont )
        {
            Device dev = Engine.g_device;

            // empty the list

            //Performance.BeginEvent(new Color4(System.Drawing.Color.Turquoise), "Set Buffers");
            

            /// set the layout to the engine
            devCont.InputAssembler.InputLayout = m_input_layout;

            // set the type of primitive(triangleList)
            devCont.InputAssembler.PrimitiveTopology = PrimitiveTopology.TriangleList;

            // set the vertex buffer 
            devCont.InputAssembler.SetVertexBuffers( 0, m_VertexBufferBinding );

            // set the index buffer
            //     dev.ImmediateContext.InputAssembler.SetIndexBuffer(m_BufferIndices, Format.R32_UInt, 0);   

            /// set the position of the mesh base on the world matrix
            /// all the change pass to the shader
            m_shader.SetThePositions(m_WorldMatrix);

            //Performance.EndEvent();

            //Performance.BeginEvent(new Color4(System.Drawing.Color.Tomato), "Execute Shaders");
            /// Execute all the passes of the shader
            m_shader.Execute( devCont );
            //Performance.EndEvent();


            //Performance.BeginEvent(new Color4(System.Drawing.Color.PaleVioletRed), "Draw");
            // Render
            devCont.Draw( m_polygons.Count * 3, 0 );
            //Performance.EndEvent();

        }



        #endregion

    }
}
