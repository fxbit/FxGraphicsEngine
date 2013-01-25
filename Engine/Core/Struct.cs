
using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;
using System.Runtime.InteropServices;
using System.Drawing;

using SlimDX;
using SlimDX.Direct3D11;

namespace GraphicsEngine.Core {
    /// <summary>
    /// Data structure that represents a message 
    /// sent from the OS to the program.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct Message {
        public IntPtr hWnd;
        public uint msg;
        public IntPtr wParam;
        public IntPtr lParam;
        public uint time;
        public Point p;
    }


    /// <summary>
    /// Represents a vertex
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct Vertex : IEquatable<Vertex> {
        // size = 4*3*4+1*2*4 = 56 bytes
        #region Variables
        public Vector3 position;
        public Vector3 normal;
        public Vector2 textCoords;
        public Vector3 tangent;
        public Vector3 binormal;
        #endregion

        #region Constructor
        /// <summary>
        /// Vertex constructor
        /// </summary>
        /// <param name="p_x">Position.X</param>
        /// <param name="p_y">Position.Y</param>
        /// <param name="p_z">Position.Z</param>
        /// <param name="n_x">Normal.X</param>
        /// <param name="n_y">Normal.Y</param>
        /// <param name="n_z">Normal.Z</param>
        /// <param name="u">TextureCoords.U</param>
        /// <param name="v">TextureCoords.V</param>
        public Vertex(float p_x, float p_y, float p_z, float n_x, float n_y, float n_z, float u, float v)
        {
            position = new Vector3(p_x, p_y, p_z);
            normal = new Vector3(n_x, n_y, n_z);
            textCoords = new Vector2(u, v);
            tangent = new Vector3();
            binormal = new Vector3();
        }
        /// <summary>
        /// Vertex constructor
        /// </summary>
        /// <param name="p">Position vector</param>
        /// <param name="n">Normal vector</param>
        /// <param name="uv">TexCoords vectors</param>
        public Vertex(Vector3 p, Vector3 n, Vector2 uv)
        {
            position = p;
            normal = n;
            textCoords = uv;
            tangent = new Vector3();
            binormal = new Vector3();
        }

        /// <summary>
        /// <summary>
        /// Vertex constructor
        /// </summary>
        /// <param name="p">Position vector</param>
        /// <param name="n">Normal vector</param>
        /// <param name="t">Tangent vector</param>
        /// <param name="b">Binormal vector</param>
        /// <param name="uv">TexCoords vectors</param>
        public Vertex(Vector3 p, Vector3 n, Vector3 t, Vector3 b, Vector2 uv)
        {
            position = p;
            normal = n;
            tangent = t;
            binormal = b;
            textCoords = uv;
        }
        #endregion

        public override bool Equals(object obj)
        {
            return Equals((Vertex)obj);
        }

        public bool Equals(Vertex obj)
        {
            return (obj.position == position) && (obj.normal == normal) && (obj.textCoords == textCoords)
                    && (obj.tangent == tangent) && (obj.binormal == binormal);
        }
    }

    /// <summary>
    /// Represents a triangle (three vertices)
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct Polygon : IEquatable<Polygon> {
        #region Variables
        public Vertex a, b, c;
        #endregion

        #region Properties
        public Vertex this[int i]
        {
            get
            {
                switch (i % 3) {
                    case 0: return a;
                    case 1: return b;
                    case 2: return c;
                }
                return a;
            }
            set
            {
                switch (i % 3) {
                    case 0: a = value; break;
                    case 1: b = value; break;
                    case 2: c = value; break;
                }
            }
        }
        #endregion

        #region Constructor
        public Polygon(Vertex a, Vertex b, Vertex c)
        {
            this.a = a;
            this.b = b;
            this.c = c;
        }
        #endregion

        public override bool Equals(object obj)
        {
            return Equals((Polygon)obj);
        }

        public bool Equals(Polygon obj)
        {
            return obj.a.Equals(this.a) && obj.b.Equals(this.b) && obj.c.Equals(this.c);
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Line {
        public Vector3 Start;
        public Vector3 End;

        public Line(Vector3 Start, Vector3 End)
        {
            this.Start = Start;
            this.End = End;
        }
    }

    public enum TextureType { Diffuse, Bump, Normal, Lightmap, VertexDisplacement, Environment , Heightmap};
    public enum ShaderViariables { Diffuse, Ambient, Specular, SpecularPower };

    public class Texture {
        #region Variables
        /// <summary>
        /// The texture file full Path name
        /// </summary>
        protected string _path;
        /// <summary>
        ///  Texture scale factor
        /// </summary>
        public float ScaleU;
        /// <summary>
        ///  Texture scale factor
        /// </summary>
        public float ScaleV;
        /// <summary>
        /// Alpha map , transparency
        /// </summary>
        private float _Alpha;
        /// <summary>
        /// Resource of texture
        /// </summary>
        public Texture2D texture2D;
        /// <summary>
        /// Resource of the shader
        /// </summary>
        public ShaderResourceView shaderResource;
        #endregion

        #region properties
        /// <summary>
        /// The path on disk where the texture is saved
        /// </summary>
        public string Path
        {
            get { return _path; }
            set { _path = value; }
        }
        /// <summary>
        /// The transparancy value
        /// </summary>
        public float Alpha
        {
            get { return _Alpha; }
            set { _Alpha = value; }
        }
        #endregion
    }

}