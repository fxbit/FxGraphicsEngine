using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualStudio.DebuggerVisualizers;
using System.Windows.Forms;
using FxMaths.Matrix;
using System.Drawing;
using FxMaths.GUI;

[assembly: System.Diagnostics.DebuggerVisualizer(
        typeof(DebugVisualizers.FxMaths.DebugFxMatrixF),
        typeof(Microsoft.VisualStudio.DebuggerVisualizers.VisualizerObjectSource),
        Target = typeof(FxMaths.Matrix.FxMatrixF),
        Description = "My First Visualizer")]
namespace DebugVisualizers.FxMaths
{
    public class DebugFxMatrixF : DialogDebuggerVisualizer
    {

        protected override void Show(IDialogVisualizerService windowService, IVisualizerObjectProvider objectProvider)
        {
            FxMatrixF mat = (FxMatrixF)objectProvider.GetObject();
            Form form = new Form();
            form.Text = string.Format("Width: {0}, Height: {1}",
                                     mat.Width, mat.Height);
            form.ClientSize = new Size(mat.Width, mat.Height);
            form.FormBorderStyle = FormBorderStyle.FixedToolWindow;

            Canvas canvas = new Canvas();
            canvas.Dock = System.Windows.Forms.DockStyle.Fill;
            canvas.Location = new System.Drawing.Point(0, 0);
            canvas.Margin = new System.Windows.Forms.Padding(4);
            canvas.Name = "canvas1";
            canvas.Size = new System.Drawing.Size(669, 513);
            canvas.TabIndex = 0;
            canvas.Zoom = new System.Drawing.SizeF(1F, 1F);

            ImageElement im = new ImageElement(mat);
            canvas.AddElements(im);

            form.Controls.Add(canvas);
            canvas.FitView();
            canvas.ReDraw();

            
            windowService.ShowDialog(form);
            //MessageBox.Show(mat[0].ToString());
        }

        public static void TestShowVisualizer(object objectToVisualize)
        {
            VisualizerDevelopmentHost visualizerHost = new VisualizerDevelopmentHost(objectToVisualize, typeof(DebugFxMatrixF));
            visualizerHost.ShowVisualizer();
        }
    }
}
