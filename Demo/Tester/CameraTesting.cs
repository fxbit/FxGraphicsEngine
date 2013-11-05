//#define USE_BGR

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
using System.Diagnostics;
using FxMaths;

namespace Tester
{
    public partial class CameraTesting : DockContent
    {
        public CameraTesting()
        {
            InitializeComponent();

        }



        #region Live camera testing


        Capture capture;

        CameraConfigs cameraConfigs = new CameraConfigs();

#if USE_BGR
        Image<Bgr, byte> nextFrame;
#else
        Image<Gray, byte> nextFrame;
#endif

        Thread captureThread = null;
        FxImages im;
        ImageElement imEl;
        ImageElement imAv;
        Boolean _running = false;

        FxMatrixF average = null;
        FxMatrixF nextMat = null;

        // fps measure
        int counts = 0;
        System.Windows.Forms.Timer fpsTimer;
        Stopwatch watch = new Stopwatch();

        private void toolStripButton1_Click(object sender, EventArgs e)
        {

            // init the camera capturing
            capture = new Capture(0);
            capture.SetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_AUTO_EXPOSURE, 1);
            //  capture.SetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_MODE, 1);
            //  capture.SetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_FPS, 60);
            //  capture.SetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_FRAME_WIDTH, 1920);
            //  capture.SetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_FRAME_HEIGHT, 1080);
            Console.WriteLine("Width:" + capture.GetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_FRAME_WIDTH));
            Console.WriteLine("Height:" + capture.GetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_FRAME_HEIGHT));
#if USE_BGR
            nextFrame = capture.QueryFrame();
#else
            nextFrame = capture.QueryGrayFrame();
#endif
            nextMat = FxMatrixF.Load(nextFrame.Bytes,
                                     nextFrame.Width,
                                     nextFrame.Height,
                                     FxMaths.Matrix.ColorSpace.Grayscale);
            average = nextMat.Copy() as FxMatrixF;

            // create the capture thread
            captureThread = new Thread(CaptureCam);

            // add the timer for the fps measure
            fpsTimer = new System.Windows.Forms.Timer();
            fpsTimer.Interval = 1000;
            watch.Start();

            fpsTimer.Tick += (s, te) => {
                watch.Stop();
                float fps = counts * 1000.0f / watch.ElapsedMilliseconds;
                m_fps.Text = fps.ToString();
                counts = 0;
                watch.Reset();
                watch.Start();
            };

            im = FxTools.FxImages_safe_constructors(nextFrame.ToBitmap());
            imEl = new ImageElement(im);
            imAv = new ImageElement(im);

            canvas1.AddElements(imEl);
            canvas1.AddElements(imAv);
            imAv.Position = new FxMaths.Vector.FxVector2f(nextFrame.Width, 0);

            _running = true;
            captureThread.Start();
            fpsTimer.Start();

            propertyGrid1.SelectedObject = cameraConfigs;
        }

        FxMatrixMask G;
        FxMatrixF s,m;

        private void CaptureCam()
        {

            G = new FxMatrixMask(nextMat.Width, nextMat.Height, false);
            s = new FxMatrixF(nextMat.Width, nextMat.Height, 1f);
            m = nextMat;

            while(_running) {

                if(nextFrame != null)
                    nextFrame.Dispose();

#if USE_BGR
                nextFrame = capture.QueryFrame();
#else
                nextFrame = capture.QueryGrayFrame();
#endif
                nextMat.Load(nextFrame.Bytes, FxMaths.Matrix.ColorSpace.Grayscale);

                var diff = nextMat - m;
                s = (cameraConfigs.a + G * (cameraConfigs.b - cameraConfigs.a)) * (diff * diff - s) + s;
                m = (cameraConfigs.a + G * (cameraConfigs.b - cameraConfigs.a)) * diff + m;
                G = s > 40;

                FxMatrixF r = s.Copy();
                r.Subtract(r.Min());
                r.Divide(r.Max());

                nextMat.Subtract(nextMat.Min());
                nextMat.Divide(nextMat.Max());

                for(int i = 0; i < 256; i++) {
                    for(int j = 0; j < 20; j++)
                        nextMat[i, j] = (i % 256) / 255.0f;
                }
                
                var grad = nextMat.Gradient(cameraConfigs.edgeDetect);
                grad.Divide(grad.Max());

                imEl.UpdateInternalImage(grad, cameraConfigs.camFrameMap);

                var mask = grad>0.1f;
                imAv.UpdateInternalImage(mask.ToFxMatrixF(), cameraConfigs.camResultMap);

                counts++;
                canvas1.ReDraw();
            }

        }

        private void CameraTesting_FormClosing(object sender, FormClosingEventArgs e)
        {
            if(captureThread != null) {
                _running = false;
                captureThread.Abort();
                captureThread.Join();
            }
        }



        #endregion


        #region Static image testing 

        private void toolStripButton2_Click(object sender, EventArgs e)
        {
            Bitmap circles = new Bitmap(@"C:\Dropbox\CHT_Grd_Scrshot.jpg");
            //Bitmap circles = new Bitmap(@"C:\Dropbox\Camera Uploads\2008-08-14 21.12.42-2.jpg");
            
            var mat = FxMatrixF.Load(circles, FxMaths.Matrix.ColorSpace.Grayscale);
            var grad = mat.Gradient(FxMatrixF.GradientMethod.Sobel);
            var grad2 = mat.Gradient(FxMatrixF.GradientMethod.Scharr);

            ImageElement imEl = new ImageElement(grad);
            canvas1.AddElements(imEl);


            imEl = new ImageElement(grad2);
            canvas1.AddElements(imEl);
        } 


        #endregion
    }


    public class CameraConfigs
    {

        #region Cam Frame
        private ColorMapDefaults _CamFrame = ColorMapDefaults.DeepBlue;
        public ColorMap camFrameMap = new ColorMap(ColorMapDefaults.DeepBlue);

        public ColorMapDefaults CamFrame
        {
            get
            {
                return _CamFrame;
            }
            set
            {
                _CamFrame = value;
                camFrameMap = new ColorMap(value);
            }
        }

        
        #endregion


        #region Cam Result
        private ColorMapDefaults _CamResult = ColorMapDefaults.Jet;
        public ColorMap camResultMap = new ColorMap(ColorMapDefaults.Jet);

        public ColorMapDefaults CamResult
        {
            get
            {
                return _CamResult;
            }
            set
            {
                _CamResult = value;
                camResultMap = new ColorMap(value);
            }
        } 
        #endregion

        public float a { get; set; }
        public float b { get; set; }
        
        public FxMatrixF.GradientMethod edgeDetect { get; set; }


        public CameraConfigs()
        {
            a = 0.5f;
            b = 0.1f;
            edgeDetect = FxMatrixF.GradientMethod.Sobel;
        }
    }
}
