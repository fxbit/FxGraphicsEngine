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
using FxMaths.Matrix;


namespace Tester
{
    public partial class CameraTesting : DockContent
    {
        Capture capture;
        Image<Bgr, byte> nextFrame;
        Thread captureThread;
        FxImages im;
        ImageElement imEl;
        ImageElement imAv;

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
            imAv = new ImageElement(im);

            canvas1.AddElements(imEl);
            canvas1.AddElements(imAv);
            imAv.Position= new FxMaths.Vector.FxVector2f(1270, 0);

            captureThread.Start();
        }

        private void processImage(FxImages im)
        {
            FxMatrixF mat = FxMatrixF.Load(im, FxMaths.Matrix.ColorSpace.Grayscale);

            mat.Multiply(0.5f);
            im.Load(mat);

        }

        FxMatrixF average = null;
        private void CaptureCam()
        {
            while (true)
            {
                if (nextFrame != null)
                    nextFrame.Dispose();

                nextFrame = capture.QueryFrame();
                FxMatrixF mat = FxMatrixF.Load(nextFrame.Bytes, nextFrame.Width, nextFrame.Height, FxMaths.Matrix.ColorSpace.Grayscale);

                if (average == null)
                {
                    average = mat;
                }
                else
                {
                    average = 0.1f*mat + 0.9f*average;
                }

            //    processImage(im);
                imEl.UpdateInternalImage(mat);
                imAv.UpdateInternalImage(average);
                
                canvas1.ReDraw();
            }

        }
    }
}
