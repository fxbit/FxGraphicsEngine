using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;


using GraphicsEngine;
using GraphicsEngine.UI;
using GraphicsEngine.Managers;

using SharpDX;
using SharpDX.Direct3D11;
using SharpDX.DXGI;
using FxMaths.Images;
using FxMaths.GUI;
using FxMaths.Vector;
using FxMaths.GMaps;
using FxMaths;
using GraphicsEngine.Core;
using GraphicsEngine.Core.PrimaryObjects3D;
using GraphicsEngine.Core.Shaders;


namespace SimpleExample
{
    public partial class Form1 : Form
    {

        #region Variables

        /// <summary>
        /// Var for direct 3d
        /// </summary>
        protected Engine engine;

        /// <summary>
        /// The viewport
        /// </summary>
        GraphicsEngine.Core.Viewport RenderArea_Viewport = null;

        #endregion


        public Form1()
        {
            InitializeComponent();
        }



        #region start 3D

        private void start3d_Click(object sender, EventArgs e)
        {
            if (!Engine.isEngineRunning)
            {

                UISettings settings = new UISettings();

                if (settings.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                {

                    // init the engine
                    //engine = new Engine(RenderArea.Width, RenderArea.Height, RenderArea.Handle, this.Handle);
                    engine = new Engine(1680, 1050, this);

                    // set the first viewport
                    RenderArea_Viewport = new GraphicsEngine.Core.Viewport(RenderArea.Width, RenderArea.Height, RenderArea.Handle, Format.R8G8B8A8_UNorm);
                    ViewportManager.AddViewport(RenderArea_Viewport);
                    /*
                    // set the second viewport
                    RenderArea_Viewport2 = new GraphicsEngine.Core.Viewport(RenderArea2.Width, RenderArea2.Height, RenderArea2.Handle, Format.R8G8B8A8_UNorm);
                    ViewportManager.AddViewport(RenderArea_Viewport2);
                    */
                    // set the moving camera
                    Engine.g_MoveCamera = RenderArea_Viewport.m_Camera;

                    // setup the input 
                    engine.SetupInput(this);

                    // start the render
                    engine.StartRender();
                }
            }
        }

        #endregion




        #region Send the mouse focus to graphic engine

        private void RenderArea_MouseClick(object sender, MouseEventArgs e)
        {
            if (engine != null && Engine.isEngineRunning)
            {
                if (e.Button == MouseButtons.Right)
                    engine.refocusInput();

                Engine.g_MoveCamera = RenderArea_Viewport.m_Camera;
            }
        }

        private void RenderArea_Resize(object sender, EventArgs e)
        {
            if (engine != null)
            {

                if (RenderArea_Viewport != null)
                    RenderArea_Viewport.Resize(RenderArea.Width, RenderArea.Height);
            }
        }


        #endregion




        #region Statistics


        private void timer1_Tick(object sender, EventArgs e)
        {
            if (engine != null)
            {
                label_fps.Text = "FPS:" + engine.FPS.ToString();
                label_triangles.Text = "Triangles:" + engine.NumTriangles;
            }
            else
            {
                label_triangles.Text = "Triangles:0";
                label_fps.Text = "FPS:0";
            }
        }
        
        #endregion




        #region Create Mesh

        private void button_create_mesh_Click(object sender, EventArgs e)
        {
            /////////////////////////////////////////////////////////////  Init the Shader

            // create the shader
            ShaderSimple sh = new ShaderSimple();

            /// add the shader to the list
            ShaderManager.AddShader("Shader10", sh);

            // add the textures for the shader
            sh.SetTexture("Resources/diffuse.jpg", TextureType.Diffuse);
            sh.SetTexture("Resources/lightmap.jpg", TextureType.Lightmap);
            sh.SetTexture("Resources/height.jpg", TextureType.Heightmap);

            /////////////////////////////////////////////////////////////  Init the polygons of mesh

            List<Polygon> polygonList = new List<Polygon>();

            FxVector2f p1 = new FxVector2f(0,0);
            FxVector2f p2 = new FxVector2f(0,100);
            FxVector2f p3 = new FxVector2f(100,100);
            FxVector2f p4 = new FxVector2f(100, 0);

            float u = 0;
            float v = 0;
            Vertex ver1 = new Vertex(p1.X, 1, p1.Y, 0, 0, 0, u, v);
            u = 0; v = 1;
            Vertex ver2 = new Vertex(p2.X, 1, p2.Y, 0, 0, 0, u, v);
            u = 1; v = 1;
            Vertex ver3 = new Vertex(p3.X, 1, p3.Y, 0, 0, 0, u, v);
            u = 1; v = 0;
            Vertex ver4 = new Vertex(p4.X, 1, p4.Y, 0, 0, 0, u, v);

            polygonList.Add(new Polygon(ver1, ver2, ver3));
            polygonList.Add(new Polygon(ver1, ver3, ver4));

            /////////////////////////////////////////////////////////////  Init the Mesh

            /// make a new mesh
            Mesh mesh = new Mesh();

            /// set to the new mesh the shader 
            mesh.m_shader = ShaderManager.GetExistShader("Shader10");

            // set the position
            mesh.SetPosition(new Vector3());

            // scale it
            mesh.SetScale(new Vector3(10, 10, 5));

            // add the polygons on mesh
            foreach (Polygon poly in polygonList){
                // add the polygons to the mesh
                mesh.AddPolygon(poly, false);
            }

            /// create the mesh and download it to the card
            mesh.CreateMesh();

            /// add the mesh to the engine mesh list
            Engine.g_MeshManager.AddMesh(mesh);


            /////////////////////////////////////////////////////////////  Change Camera position

            Engine.g_MoveCamera.SetViewParams(new Vector3(2000, 2000, 200),
                                              new Vector3(0, 0, 0));
        }
        
        #endregion



    }
}
