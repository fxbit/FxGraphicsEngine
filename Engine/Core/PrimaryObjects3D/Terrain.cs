using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using SlimDX;
using FxMaths.Noise;

namespace GraphicsEngine.Core.PrimaryObjects3D
{
    public class Terrain : Mesh
    {
        #region Private Variables

        private float m_width;
        
        private float m_height;

        #endregion

        #region Properties

        /// <summary>
        /// Get the Width of the terrain
        /// </summary>
        public float Width { get { return m_width; } }

        /// <summary>
        /// Get the Height of the terrain
        /// </summary>
        public float Height { get { return m_height; } }

        #endregion


        public Terrain(float width, float height, int divisions , INoise2D noise) : base()
        {
            // init the internal variables
            m_width = width;
            m_height = height;

            // calc the number of the vertex in the grid
            int numVertex = (divisions + 1) * (divisions + 1);

            // init the poligons of the terrain
            Vertex[,] verts = new Vertex[(divisions + 1), (divisions + 1)];

            // calc the positions of the vertex
            for (int y = 0; y <= divisions; y++)
            {
                for (int x = 0; x <= divisions; x++)
                {
                    float p_x=(float)x/(divisions+1)*width;
                    float p_z=(float)y/(divisions+1)*height;
                    
                    float u=(float)x/(divisions+1);
                    float v=(float)y/(divisions+1);
                    float p_y = noise.GetValue( v, u );

                    verts[x,y] = new Vertex(p_x, p_y, p_z, 0, 0, 0, v, u);
                }
            }

            for (int y = 0; y < divisions; y++)
            {
                for (int x = 0; x < divisions; x++)
                {
                    // add the polygons
                    this.AddPolygon(verts[x,y], verts[x+1,y], verts[x,y+1], false);
                    this.AddPolygon(verts[x,y+1], verts[x+1,y+1], verts[x+1,y], false);
                }
            }

        }
    }
}
