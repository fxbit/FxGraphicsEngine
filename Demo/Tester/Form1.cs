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
using SharpDX.MediaFoundation;
using System.IO;
using FXFramework;


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



        MediaPlayer mp = new MediaPlayer();
        Texture difTex;


        public TesterForm()
        {
            InitializeComponent();

            // init the console
            UIConsole = new ConsoleOutput();
            UIConsole.Show(dockPanel1, DockState.DockBottom);
            outputToolStripMenuItem.Checked = true;
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
            if (outputToolStripMenuItem.Checked)
            {
                UIConsole.Hide();
                outputToolStripMenuItem.Checked = false;
            }
            else
            {
                if (UIConsole == null)
                {
                    UIConsole = new ConsoleOutput();

                    // add the viewport to the dock
                    UIConsole.Show(dockPanel1, DockState.DockBottom);
                }
                else
                {
                    // add the viewport to the dock
                    UIConsole.Show(dockPanel1, DockState.DockBottom);
                }
                outputToolStripMenuItem.Checked = true;
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
            sh.SetVariables(new Vector3(1, 1, 1), ShaderViariables.Ambient);

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







        #region Video Plane
        


        private void addVideoPlaneToolStripMenuItem_Click(object sender, EventArgs e)
        {
            /////////////////////////////////////////////////////////////  Start Video
            mp.Initialize(Engine.g_device);

            mp.SetFile("sample.avi");
            difTex = mp.CreateTexture();

            Engine.AppHandleKeys = (SharpDX.DirectInput.Key key) =>
            {
                /// handle the keys that we want
                switch (key)
                {
                    case SharpDX.DirectInput.Key.Up:
                        mp.OnRender(difTex.texture2D, true);
                        break;
                    case SharpDX.DirectInput.Key.Down:
                        mp.OnRender(difTex.texture2D, false);
                        break;
                }

                return false;
            };

            /////////////////////////////////////////////////////////////  Init the Shader

            // create the shader
            ShaderSimple sh = new ShaderSimple();

            /// add the shader to the list
            ShaderManager.AddShader("Shader10", sh);

            // add the textures for the shader
            sh.SetTexture(difTex, TextureType.Diffuse);
            sh.SetTexture(difTex, TextureType.Lightmap);
            sh.SetTexture("Resources/height.jpg", TextureType.Heightmap);
            sh.SetVariables(new Vector3(1, 1, 1), ShaderViariables.Ambient);

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







        #region Sphere

        public class sphere_info
        {
            public ShaderSimple sh;
            public Vector3 center;
            public float r;
        }

        // create the shader
        List<sphere_info> listSphere = new List<sphere_info>();


        private void AddSphete(float r, Vector3 center, Vector3 scale)
        {

            ShaderSimple sh = new ShaderSimple();


            // add to list the sphere
            sphere_info spi = new sphere_info();
            spi.sh = sh;
            spi.r = r * scale.X * 1.5f;
            spi.center = center;
            listSphere.Add(spi);


            // add the textures for the shader
            sh.SetTexture(difTex, TextureType.Diffuse);
            sh.SetTexture("Resources/white.jpg", TextureType.Lightmap);
            sh.SetTexture("Resources/height.jpg", TextureType.Heightmap);


            ShaderManager.AddShader("Shader" + r.ToString(), sh);

            List<Polygon> polygonList = new List<Polygon>();

            float numSteps = 100;
            float thita_step = (float)(Math.PI / numSteps);
            float phi_step = (float)(2 * Math.PI / numSteps);
            for (float thita = 0; thita < Math.PI; thita += thita_step)
            {
                for (float phi = 0; phi < 2 * Math.PI; phi += phi_step)
                {
                    float x = (float)(r * Math.Sin(thita) * Math.Cos(phi));
                    float y = (float)(r * Math.Sin(thita) * Math.Sin(phi));
                    float z = (float)(r * Math.Cos(thita));
                    float u = (float)(thita / (Math.PI));
                    float v = (float)(phi / (2 * Math.PI));
                    Vertex ver1A = new Vertex(x, z, y, 0, 0, 0, v, u);

                    x = (float)(r * Math.Sin(thita) * Math.Cos(phi + phi_step));
                    y = (float)(r * Math.Sin(thita) * Math.Sin(phi + phi_step));
                    z = (float)(r * Math.Cos(thita));
                    u = (float)(thita / (Math.PI));
                    v = (float)((phi + phi_step) / (2 * Math.PI));
                    Vertex ver2A = new Vertex(x, z, y, 0, 0, 0, v, u);

                    x = (float)(r * Math.Sin(thita + thita_step) * Math.Cos(phi + phi_step));
                    y = (float)(r * Math.Sin(thita + thita_step) * Math.Sin(phi + phi_step));
                    z = (float)(r * Math.Cos(thita + thita_step));
                    u = (float)((thita + thita_step) / (Math.PI));
                    v = (float)((phi + phi_step) / (2 * Math.PI));
                    Vertex ver3A = new Vertex(x, z, y, 0, 0, 0, v, u);

                    x = (float)(r * Math.Sin(thita + thita_step) * Math.Cos(phi));
                    y = (float)(r * Math.Sin(thita + thita_step) * Math.Sin(phi));
                    z = (float)(r * Math.Cos(thita + thita_step));
                    u = (float)((thita + thita_step) / (Math.PI));
                    v = (float)((phi) / (2 * Math.PI));
                    Vertex ver4A = new Vertex(x, z, y, 0, 0, 0, v, u);


                    polygonList.Add(new Polygon(ver1A, ver2A, ver3A));
                    polygonList.Add(new Polygon(ver1A, ver3A, ver4A));
                }
            }



            Mesh mesh = new Mesh();
            /// set to the new mesh the shader 
            mesh.m_shader = sh;
            // set the position
            mesh.SetPosition(center);
            // scale it
            mesh.SetScale(scale);


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


            sh.SetVariables(new Vector3(1, 1, 1), ShaderViariables.Ambient);
        }


        private void AddSphere(float r, Vector3 center, Vector3 scale, String imagePath)
        {

            ShaderSimple sh = new ShaderSimple();

            // add to list the sphere
            sphere_info spi = new sphere_info();
            spi.sh = sh;
            spi.r = r * scale.X * 1.5f;
            spi.center = center;
            listSphere.Add(spi);


            // add the textures for the shader
            sh.SetTexture(imagePath, TextureType.Diffuse);
            sh.SetTexture("Resources/white.jpg", TextureType.Lightmap);
            sh.SetTexture("Resources/height.jpg", TextureType.Heightmap);


            ShaderManager.AddShader("Shader" + r.ToString(), sh);

            List<Polygon> polygonList = new List<Polygon>();

            float numSteps = 100;
            float thita_step = (float)(Math.PI / numSteps);
            float phi_step = (float)(2 * Math.PI / numSteps);
            for (float thita = 0; thita < Math.PI; thita += thita_step)
            {
                for (float phi = 0; phi < 2 * Math.PI; phi += phi_step)
                {
                    float x = (float)(r * Math.Sin(thita) * Math.Cos(phi));
                    float y = (float)(r * Math.Sin(thita) * Math.Sin(phi));
                    float z = (float)(r * Math.Cos(thita));
                    float u = (float)(thita / (Math.PI));
                    float v = (float)(phi / (2 * Math.PI));
                    Vertex ver1A = new Vertex(x, z, y, 0, 0, 0, v, u);

                    x = (float)(r * Math.Sin(thita) * Math.Cos(phi + phi_step));
                    y = (float)(r * Math.Sin(thita) * Math.Sin(phi + phi_step));
                    z = (float)(r * Math.Cos(thita));
                    u = (float)(thita / (Math.PI));
                    v = (float)((phi + phi_step) / (2 * Math.PI));
                    Vertex ver2A = new Vertex(x, z, y, 0, 0, 0, v, u);

                    x = (float)(r * Math.Sin(thita + thita_step) * Math.Cos(phi + phi_step));
                    y = (float)(r * Math.Sin(thita + thita_step) * Math.Sin(phi + phi_step));
                    z = (float)(r * Math.Cos(thita + thita_step));
                    u = (float)((thita + thita_step) / (Math.PI));
                    v = (float)((phi + phi_step) / (2 * Math.PI));
                    Vertex ver3A = new Vertex(x, z, y, 0, 0, 0, v, u);

                    x = (float)(r * Math.Sin(thita + thita_step) * Math.Cos(phi));
                    y = (float)(r * Math.Sin(thita + thita_step) * Math.Sin(phi));
                    z = (float)(r * Math.Cos(thita + thita_step));
                    u = (float)((thita + thita_step) / (Math.PI));
                    v = (float)((phi) / (2 * Math.PI));
                    Vertex ver4A = new Vertex(x, z, y, 0, 0, 0, v, u);


                    polygonList.Add(new Polygon(ver1A, ver2A, ver3A));
                    polygonList.Add(new Polygon(ver1A, ver3A, ver4A));
                }
            }



            Mesh mesh = new Mesh();
            /// set to the new mesh the shader 
            mesh.m_shader = sh;
            // set the position
            mesh.SetPosition(center);
            // scale it
            mesh.SetScale(scale);


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


            sh.SetVariables(new Vector3(1, 1, 1), ShaderViariables.Ambient);
        }
        #endregion








        #region Video Sphere
        


        private void addVideoSphereToolStripMenuItem_Click(object sender, EventArgs e)
        {
            /////////////////////////////////////////////////////////////  Start Video

            mp.Initialize(Engine.g_device);

            mp.SetFile("sample.avi");
            difTex = mp.CreateTexture();


            Engine.AppHandleKeys = (SharpDX.DirectInput.Key key) =>
            {
                /// handle the keys that we want
                switch (key)
                {
                    case SharpDX.DirectInput.Key.Up:
                        mp.OnRender(difTex.texture2D, true);
                        break;
                    case SharpDX.DirectInput.Key.Down:
                        mp.OnRender(difTex.texture2D, false);
                        break;
                }

                return false;
            };


            /////////////////////////////////////////////////////////////  Add the Spheres

            AddSphete(1000, new Vector3(0, 0, 0), new Vector3(20, 20, 20));
            //AddSphete(100, new Vector3(1100, 80, 2000), new Vector3(3, 3, 3));

            /////////////////////////////////////////////////////////////  Init the Shader
#if false

            // create the shader
            ShaderSimple sh = new ShaderSimple();

            /// add the shader to the list
            ShaderManager.AddShader("Shader12", sh);

            // add the textures for the shader
            sh.SetTexture("Resources/diffuse.jpg", TextureType.Diffuse);
            sh.SetTexture("Resources/lightmap.jpg", TextureType.Lightmap);
            sh.SetTexture("Resources/height.jpg", TextureType.Heightmap);
            sh.SetVariables(new Vector3(1, 1, 1), ShaderViariables.Ambient);

            List<Polygon> polygonList = new List<Polygon>();

            FxVector2f p1 = new FxVector2f(0, 0);
            FxVector2f p2 = new FxVector2f(0, 100);
            FxVector2f p3 = new FxVector2f(100, 100);
            FxVector2f p4 = new FxVector2f(100, 0);

            float u1 = 0;
            float v1 = 0;
            Vertex ver1 = new Vertex(p1.X, -1, p1.Y, 0, 0, 0, u1, v1);
            u1 = 0; v1 = 1;
            Vertex ver2 = new Vertex(p2.X, -1, p2.Y, 0, 0, 0, u1, v1);
            u1 = 1; v1 = 1;
            Vertex ver3 = new Vertex(p3.X, -1, p3.Y, 0, 0, 0, u1, v1);
            u1 = 1; v1 = 0;
            Vertex ver4 = new Vertex(p4.X, -1, p4.Y, 0, 0, 0, u1, v1);

            polygonList.Add(new Polygon(ver1, ver2, ver3));
            polygonList.Add(new Polygon(ver1, ver3, ver4));

            /////////////////////////////////////////////////////////////  Init the Mesh

            /// make a new mesh
            Mesh mesh = new Mesh();
            /// set to the new mesh the shader 
            mesh.m_shader = ShaderManager.GetExistShader("Shader12");
            // set the position
            mesh.SetPosition(new Vector3(0, 0, 0));

            // scale it
            mesh.SetScale(new Vector3(40, 40, 40));

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

            sh.SetVariables(new Vector3(1, 1, 1), ShaderViariables.Ambient);

            
#endif

            /////////////////////////////////////////////////////////////  Change Camera position
            // Engine.g_MoveCamera.
            Engine.g_MoveCamera.SetViewParams(new Vector3(2000, 2000, 200),
                                              new Vector3(0, 0, 0));


            ////////////////////////////////////////////////////
        }

        #endregion






        #region Add CEID

        sphere_info selected_sphere = null;
        Boolean weAreIn = false;
        private void HitTest_Tick(object sender, EventArgs e)
        {
            if (!weAreIn)
            {
                // clean the high light from all spheres
                foreach (sphere_info spi in listSphere)
                {
                    spi.sh.SetVariables(new Vector3(1, 1, 1), ShaderViariables.Ambient);
                }


                // try to find a hit
                foreach (sphere_info spi in listSphere)
                {
                    Vector3 x0 = spi.center;
                    Vector3 x1 = Engine.g_MoveCamera.LookAt;
                    Vector3 x2 = Engine.g_MoveCamera.Eye;
                    Vector3 sub1 = Vector3.Subtract(x0, x1);
                    Vector3 sub2 = Vector3.Subtract(x0, x2);
                    Vector3 sub3 = Vector3.Subtract(x2, x1);

                    Vector3 d_arith = Vector3.Cross(sub1, sub2);
                    float dis = (d_arith.Length() / sub3.Length());

                    if (dis < spi.r)
                    {
                        spi.sh.SetVariables(new Vector3(1.0f, 0.5f, 0.5f), ShaderViariables.Ambient);
                        selected_sphere = spi;
                        return;
                    }
                }


                selected_sphere = null;
            }
        }

        private void addCEIDToolStripMenuItem_Click(object sender, EventArgs e)
        {
            /////////////////////////////////////////////////////////////  Start Video

            mp.Initialize(Engine.g_device);

            mp.SetFile("sample.avi");
            difTex = mp.CreateTexture();


            Engine.AppHandleKeys = (SharpDX.DirectInput.Key key) =>
            {
                /// handle the keys that we want
                switch (key)
                {
                    case SharpDX.DirectInput.Key.Up:
                        mp.OnRender(difTex.texture2D, true);
                        break;
                    case SharpDX.DirectInput.Key.Down:
                        mp.OnRender(difTex.texture2D, false);
                        break;
                    case SharpDX.DirectInput.Key.Space:
                        if (selected_sphere != null)
                        {
                            Engine.g_MoveCamera.SetViewParams(selected_sphere.center,
                                          new Vector3(2000, 0, 2000));

                            selected_sphere.sh.SetVariables(new Vector3(1, 1, 1), ShaderViariables.Ambient);

                            selected_sphere = null;
                            weAreIn = true;
                        }
                        break;
                    case SharpDX.DirectInput.Key.LeftShift:
                        Engine.g_MoveCamera.SetViewParams(new Vector3(6500, 4500, 2000),
                              new Vector3(200, 0, 200));
                        weAreIn = false;
                        break;
                }

                return false;
            };

            /////////////////////////////////////////////////////////////  Add the Spheres

            //  AddSphete(1000, new Vector3(0, 0, 0), new Vector3(2, 2, 2),"sample.avi");
            AddSphere(100, new Vector3(1500, 110, 900), new Vector3(2.5f, 2.5f, 2.5f), "Resources/lady2.jpg");
            AddSphere(100, new Vector3(2300, 80, 3000), new Vector3(2.5f, 2.5f, 2.5f), "Resources/lady.jpg");
            AddSphere(100, new Vector3(2300, 110, 2300), new Vector3(2.5f, 2.5f, 2.5f), "Resources/lady3.jpg");
            AddSphete(100, new Vector3(2300, 110, 1700), new Vector3(2.5f, 2.5f, 2.5f));
            // AddSphete(100, new Vector3(1100, 80, 2000), new Vector3(3, 3, 3));


            // create the shader
            ShaderSimple sh1 = new ShaderSimple();

            /// add the shader to the list
            ShaderManager.AddShader("Shader11", sh1);

            // add the textures for the shader
            sh1.SetTexture("Resources/tmima2.png", TextureType.Diffuse);
            sh1.SetTexture("Resources/tmima2.png", TextureType.Lightmap);
            sh1.SetTexture("Resources/height.jpg", TextureType.Heightmap);


            List<Polygon> polygonList1 = new List<Polygon>();

            FxVector2f p1 = new FxVector2f(0, 0);
            FxVector2f p2 = new FxVector2f(0, 100);
            FxVector2f p3 = new FxVector2f(100, 100);
            FxVector2f p4 = new FxVector2f(100, 0);

            float u1 = 0;
            float v1 = 0;
            Vertex ver1 = new Vertex(p1.X, -1, p1.Y, 0, 0, 0, u1, v1);
            u1 = 0; v1 = 1;
            Vertex ver2 = new Vertex(p2.X, -1, p2.Y, 0, 0, 0, u1, v1);
            u1 = 1; v1 = 1;
            Vertex ver3 = new Vertex(p3.X, -1, p3.Y, 0, 0, 0, u1, v1);
            u1 = 1; v1 = 0;
            Vertex ver4 = new Vertex(p4.X, -1, p4.Y, 0, 0, 0, u1, v1);

            polygonList1.Add(new Polygon(ver1, ver2, ver3));
            polygonList1.Add(new Polygon(ver1, ver3, ver4));


            /// make a new mesh
            Mesh mesh1 = new Mesh();
            /// set to the new mesh the shader 
            mesh1.m_shader = ShaderManager.GetExistShader("Shader11");
            // set the position
            mesh1.SetPosition(new Vector3(0, 0, 0));

            // scale it
            mesh1.SetScale(new Vector3(60, 60, 60));

            // add the polygons on mesh
            foreach (Polygon poly in polygonList1)
            {
                // add the polygons to the mesh
                mesh1.AddPolygon(poly, false);
            }


            /// create the mesh and download it to the card
            mesh1.CreateMesh();

            /// add the mesh to the engine mesh list
            Engine.g_MeshManager.AddMesh(mesh1);

            sh1.SetVariables(new Vector3(1, 1, 1), ShaderViariables.Ambient);

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
            Viewport2D viewport = new Viewport2D();

            viewport.Show(dockPanel1, DockState.Document);

            // start the graphic engine
            StartGraphicEngine();

            // init  a local delaunay
            DelaunayCS delaunay = new DelaunayCS();
            
            // add a random points  TODO: add external source (ex. file)
            delaunay.CreateRandomPoints(10000, new FxVector2f(0, 0), new FxVector2f(10000, 10000));

            // init the shader part of delaunay
            delaunay.InitShaders(Engine.g_device);

            // show the points
            delaunay.DrawPoints(viewport.canvas);
            
            // run the algorithm 
            delaunay.RunTheAlgorithm(viewport.canvas);

            // show the triangles
            delaunay.DrawTriangles(viewport.canvas);
        }
        
        #endregion

        #endregion







        #region Camera 

        CameraTesting cam;
        private void addCameraViewportToolStripMenuItem_Click(object sender, EventArgs e)
        {
            // create a new viewport
            cam = new CameraTesting();

            // add the viewport to the dock
            cam.Show(dockPanel1, DockState.Document);
        }

        #endregion




        private void TesterForm_FormClosing(object sender, FormClosingEventArgs e)
        {
            if(cam != null)
                cam.Close();
        }



        #region Kinect

        private  KinectForm kinect;
        private void addKinectViewportToolStripMenuItem_Click(object sender, EventArgs e)
        {
            // create a new viewport
            kinect = new KinectForm();

            // add the viewport to the dock
            kinect.Show(dockPanel1, DockState.Document);
        } 


        #endregion

        private void testJpegToolStripMenuItem_Click(object sender, EventArgs e)
        {
            OpenFileDialog fileOpen = new OpenFileDialog();

            if(fileOpen.ShowDialog() == System.Windows.Forms.DialogResult.OK) {
                Bitmap bitMap = new Bitmap(fileOpen.FileName);
                FxImages fxIm = FxTools.FxImages_safe_constructors(bitMap);
                StringBuilder sb = new StringBuilder();
                sb.Append("byte image[] = {");
                int Width = fxIm.Image.Width;
                int Height = fxIm.Image.Height;
                fxIm.LockImage();
                for(int i=0; i < Width; i++) {
                    for(int j=0; j < Height; j++) {
                        if(j % 32 == 0)
                            sb.Append("\r\n");
                        sb.Append(fxIm[i, j, RGB.R]); sb.Append(',');
                        sb.Append(fxIm[i, j, RGB.G]); sb.Append(',');
                        sb.Append(fxIm[i, j, RGB.B]); sb.Append(',');
                    }

                }
                fxIm.UnLockImage();
                sb.Append("};");
                System.IO.File.WriteAllText("image.c", sb.ToString());
            }
        }
    }
}
