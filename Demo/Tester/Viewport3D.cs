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


namespace Tester
{
    public partial class Viewport3D : DockContent
    {

        #region Variables

        Engine engine;

        /// <summary>
        /// The viewport
        /// </summary>
        GraphicsEngine.Core.Viewport RenderArea_Viewport = null;

        #endregion



        public Viewport3D(Engine engine)
        {
            InitializeComponent();

            // save localy the graphic engine
            this.engine = engine;

            // set the first viewport
            RenderArea_Viewport = new GraphicsEngine.Core.Viewport(RenderArea.Width, RenderArea.Height, RenderArea.Handle, Format.R8G8B8A8_UNorm);
            ViewportManager.AddViewport(RenderArea_Viewport);

            // set the moving camera
            Engine.g_MoveCamera = RenderArea_Viewport.m_Camera;

        }



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



        private void Viewport_FormClosing(object sender, FormClosingEventArgs e)
        {
            // remove from rendering the current viewport
            ViewportManager.RemoveViewport(RenderArea_Viewport);
        }


    }
}
