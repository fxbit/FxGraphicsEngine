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

        // create the shader
        List<sphere_info> listSphere= new List<sphere_info>();
        

        private void AddSphete(float r, Vector3 center, Vector3 scale, String imagePath)
        {
            
            ShaderSimple sh = new ShaderSimple();


            // add to list the sphere
            sphere_info spi = new sphere_info();
            spi.sh = sh;
            spi.r = r * scale.X*1.5f;
            spi.center = center;
            listSphere.Add(spi);


            // add the textures for the shader
            sh.SetTexture(imagePath, TextureType.Diffuse);
            sh.SetTexture(imagePath, TextureType.Lightmap);
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


        Boolean firstTime = true;
        Timer t = new Timer();

        private void button_create_mesh_Click(object sender, EventArgs e)
        {

            if (firstTime)
            {
                firstTime = false;
                AddSphete(100, new Vector3(600, 80, 600), new Vector3(2, 2, 2), "Resources/lady2.jpg");
                AddSphete(100, new Vector3(1100, 80, 2000), new Vector3(3, 3, 3), "Resources/lady.jpg");
                /////////////////////////////////////////////////////////////  Init the Shader

                // create the shader
                ShaderSimple sh = new ShaderSimple();

                /// add the shader to the list
                ShaderManager.AddShader("Shader12", sh);

                // add the textures for the shader
                sh.SetTexture("Resources/tmima.jpg", TextureType.Diffuse);
                sh.SetTexture("Resources/tmima.jpg", TextureType.Lightmap);
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


                /////////////////////////////////////////////////////////////  Change Camera position
                // Engine.g_MoveCamera.
                Engine.g_MoveCamera.SetViewParams(new Vector3(4500, 3500, 2000),
                                                  new Vector3(200, 0, 200));


                ////////////////////////////////////////////////////

                
                t.Interval = 200;
                t.Tick += t_Tick;
                t.Enabled = true;
            }
            else
            {
                if (selected_sphere != null)
                {
                    Engine.g_MoveCamera.SetViewParams(selected_sphere.center,
                                  new Vector3(200, 0, 200));
                    selected_sphere.sh.SetVariables(new Vector3(1, 1, 1), ShaderViariables.Ambient);
                    t.Enabled = false;

                }

            }
        }


        sphere_info selected_sphere = null;

        void t_Tick(object sender, EventArgs e)
        {
            selected_sphere = null;
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
                }
                else
                {
                    spi.sh.SetVariables(new Vector3(1, 1, 1), ShaderViariables.Ambient);

                }
            }
        }






        #endregion



    }




    public class sphere_info
    {
        public ShaderSimple sh;
        public Vector3 center;
        public float r;
    }

}

