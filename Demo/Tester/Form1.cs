using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using WeifenLuo.WinFormsUI.Docking;


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

using Delaunay;


namespace Tester
{
    public partial class TesterForm : Form
    {

        /// <summary>
        /// Var for direct 3d
        /// </summary>
        protected Engine engine;

        /// <summary>
        /// Form for debuging.
        /// You can print that console by call UIConsole.Write
        /// </summary>
        public static ConsoleOutput UIConsole = null;

        public TesterForm()
        {
            InitializeComponent();

            // init the console
            UIConsole = new ConsoleOutput();

        }



        #region Init Graphic Engine

        public void StartGraphicEngine()
        {
            if (!Engine.isEngineRunning)
            {

                UISettings settings = new UISettings();

                if (settings.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                {
                    // init the engine
                    engine = new Engine(1680, 1050, this);

                    // setup the input 
                    engine.SetupInput(this);

                    // start the render
                    engine.StartRender();
                }
            }
        }

        #endregion



        #region View Menu


        #region Add 3D ViewPort

        private void addViewportToolStripMenuItem_Click(object sender, EventArgs e)
        {
            // start the graphic engine
            StartGraphicEngine();

            // create a new viewport
            Viewport3D viewport = new Viewport3D(engine);

            // add the viewport to the dock
            viewport.Show(dockPanel1, DockState.Document);
        }

        #endregion




        #region Add 2D ViewPort

        private void add2DViewportToolStripMenuItem_Click(object sender, EventArgs e)
        {
            // create a new viewport
            Viewport2D viewport = new Viewport2D();

            // add the viewport to the dock
            viewport.Show(dockPanel1, DockState.Document);
        }

        #endregion




        #region Add Output window
        
        private void outputToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (UIConsole == null)
            {
                UIConsole = new ConsoleOutput();

                // add the viewport to the dock
                UIConsole.Show(dockPanel1, DockState.Document);
            }
            else
            {
                // add the viewport to the dock
                UIConsole.Show(dockPanel1, DockState.Document);
            }
        } 

        #endregion

        #endregion



        #region Statistics


        private void timer_statistics_update_Tick(object sender, EventArgs e)
        {
            if (engine != null)
            {
                toolStripStatusLabel_fps.Text = "FPS:" + engine.FPS.ToString();
                toolStripStatusLabel_triangles.Text = "Triangles:" + engine.NumTriangles;
            }
            else
            {
                toolStripStatusLabel_triangles.Text = "Triangles:0";
                toolStripStatusLabel_fps.Text = "FPS:0";
            }
        }

        #endregion



        #region DirectX 3D Menu


        #region Add Sample Plane


        private void addSamplePlaneToolStripMenuItem_Click(object sender, EventArgs e)
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

            FxVector2f p1 = new FxVector2f(0, 0);
            FxVector2f p2 = new FxVector2f(0, 100);
            FxVector2f p3 = new FxVector2f(100, 100);
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
            foreach (Polygon poly in polygonList)
            {
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


        #endregion





        #region DirectX 2D Menu



        #region Add image on the selected Canva
        private void addImageToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Viewport2D viewport = null;

            if (dockPanel1.ActiveContent is Viewport2D)
            {
                viewport = (Viewport2D)dockPanel1.ActiveContent;
            }
            else if (dockPanel1.ActiveDocument is Viewport2D)
            {
                viewport = (Viewport2D)dockPanel1.ActiveDocument;
            }

            if (viewport != null)
            {
                if (openFileDialog1.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                {
                    // load the image that the user have select
                    FxImages im = FxTools.FxImages_safe_constructors(new Bitmap(openFileDialog1.FileName));

                    // create a new image element 
                    ImageElement imE = new ImageElement(im);

                    // add the element on canva
                    viewport.AddElement(imE);
                }
            }
        }
        #endregion


        #endregion





        private void Form1_ResizeEnd(object sender, EventArgs e)
        {
        }




        #region DirectCompute


        #region Delaunay

        private void delaunay2DToolStripMenuItem_Click(object sender, EventArgs e)
        {
            DelaunayCS delaunay = new DelaunayCS();
            delaunay.CreateRandomPoints(100, new FxVector2f(0, 0), new FxVector2f(100, 100));

            //delaunay.InitShaders(Engine.g_device);

        }
        
        #endregion






        #endregion
    }
}
