using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

using WeifenLuo.WinFormsUI.Docking;

using FxMaths.GUI;

namespace Tester
{
    public partial class Viewport2D : DockContent
    {
        public Viewport2D()
        {
            InitializeComponent();
        }

        public void AddElement(FxMaths.GUI.CanvasElements elements)
        {
            canvas.AddElements(elements);
        }

        private void Viewport2D_ResizeEnd(object sender, EventArgs e)
        {
            canvas.ReDraw();
        }
    }
}
