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
        typeof(DebugVisualizers.FxMaths.DebugFxMatrix),
        typeof(Microsoft.VisualStudio.DebuggerVisualizers.VisualizerObjectSource),
        Target = typeof(FxMaths.Matrix.FxMatrixF),
        Description = "Image Viewing")]
[assembly: System.Diagnostics.DebuggerVisualizer(
        typeof(DebugVisualizers.FxMaths.DebugFxMatrix),
        typeof(Microsoft.VisualStudio.DebuggerVisualizers.VisualizerObjectSource),
        Target = typeof(FxMaths.Matrix.FxMatrixMask),
        Description = "Image Viewing")]
namespace DebugVisualizers.FxMaths
{
    public class DebugFxMatrix : DialogDebuggerVisualizer
    {
        protected override void Show(IDialogVisualizerService windowService, IVisualizerObjectProvider objectProvider)
        {
            object obj = objectProvider.GetObject();
            FxMatrixF mat;
            if (obj is FxMatrixF || obj is FxMatrix<float>)
                mat = obj as FxMatrixF;
            else if (obj is FxMatrixMask)
                mat = (obj as FxMatrixMask).ToFxMatrixF();
            else
                return;
             
            /*
            Form form = new Form();
            
            form.Text = string.Format("Width: {0}, Height: {1}",
                                     mat.Width, mat.Height);
            form.ClientSize = new Size(mat.Width, mat.Height);
            form.FormBorderStyle = FormBorderStyle.FixedToolWindow;
            */

            Canvas canvas = new Canvas();
            canvas.Dock = DockStyle.Fill;
            canvas.Location = new System.Drawing.Point(0, 0);
            canvas.Margin = new System.Windows.Forms.Padding(4);
            canvas.Name = "canvas1";
            canvas.Size = new System.Drawing.Size(mat.Width, mat.Height+32);
            canvas.MinimumSize = new System.Drawing.Size(mat.Width, mat.Height + 32);
            canvas.Zoom = new System.Drawing.SizeF(1F, 1F);

            
            ImageElement im = new ImageElement(mat, new global::FxMaths.Images.ColorMap(global::FxMaths.Images.ColorMapDefaults.Jet));
            canvas.AddElement(im);
            canvas.FitView();

            //form.Controls.Add(canvas);
            //form.Show();
            windowService.ShowDialog(canvas);
        }

        public static void TestShowVisualizer(object objectToVisualize)
        {
            VisualizerDevelopmentHost visualizerHost = new VisualizerDevelopmentHost(objectToVisualize, typeof(DebugFxMatrix));
            visualizerHost.ShowVisualizer();
        }
    }
}
