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
using FxMaths.Vector;

[assembly: System.Diagnostics.DebuggerVisualizer(
        typeof(DebugVisualizers.FxMaths.DebugFxVectorF),
        typeof(Microsoft.VisualStudio.DebuggerVisualizers.VisualizerObjectSource),
        Target = typeof(FxMaths.Vector.FxVectorF),
        Description = "Plot Viewing")]
namespace DebugVisualizers.FxMaths
{
    public class DebugFxVectorF : DialogDebuggerVisualizer
    {
        protected override void Show(IDialogVisualizerService windowService, IVisualizerObjectProvider objectProvider)
        {
            object obj = objectProvider.GetObject();
            FxVectorF vec;
            if (obj is FxVectorF || obj is FxVector<float>)
                vec = obj as FxVectorF;
            else
                return;
             
            Canvas canvas = new Canvas();
            canvas.Dock = DockStyle.Fill;
            canvas.Location = new System.Drawing.Point(0, 0);
            canvas.Margin = new System.Windows.Forms.Padding(4);
            canvas.Name = "canvas1";
            canvas.Zoom = new System.Drawing.SizeF(1F, 1F);


            PloterElement plot = new PloterElement(vec);
            canvas.AddElement(plot);
            canvas.FitView();

            windowService.ShowDialog(canvas);
        }

        public static void TestShowVisualizer(object objectToVisualize)
        {
            VisualizerDevelopmentHost visualizerHost = new VisualizerDevelopmentHost(objectToVisualize, typeof(DebugFxMatrix));
            visualizerHost.ShowVisualizer();
        }
    }
}
