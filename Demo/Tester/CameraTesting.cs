using Emgu.CV;
using Emgu.CV.Structure;
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
using FxMaths.Images;
using System.Threading;


namespace Tester
{
    public partial class CameraTesting : DockContent
    {
        Capture capture;
        Image<Bgr, byte> nextFrame;
        Thread captureThread;
        FxImages im;
        ImageElement imEl;

        public CameraTesting()
        {
            InitializeComponent();


            capture = new Capture(1);
            capture.SetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_AUTO_EXPOSURE, 1);
            capture.SetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_MODE, 1);
            capture.SetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_FRAME_WIDTH, 1920);
            capture.SetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_FRAME_HEIGHT, 1080);

            nextFrame = capture.QueryFrame();

            captureThread = new Thread(CaptureCam);
        }

        private void CameraTesting_Load(object sender, EventArgs e)
        {

        }

        private void toolStripButton1_Click(object sender, EventArgs e)
        {
            im = FxTools.FxImages_safe_constructors(nextFrame.ToBitmap());
            imEl = new ImageElement(im);

            canvas1.AddElements(imEl);

            captureThread.Start();
        }

        private void CaptureCam()
        {
            while (true)
            {
                if (nextFrame != null)
                    nextFrame.Dispose();

                nextFrame = capture.QueryFrame();
                im = FxTools.FxImages_safe_constructors(nextFrame.ToBitmap());
                imEl.UpdateIternalImage(im);

                canvas1.ReDraw();
            }

        }
    }
}
