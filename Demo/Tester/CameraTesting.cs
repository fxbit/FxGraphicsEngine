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
using System.Threading.Tasks;
using Emgu.CV.UI;
using Emgu.CV.VideoSurveillance;

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
            capture.SetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_FRAME_WIDTH, 640);
            capture.SetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_FRAME_HEIGHT, 480);
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

            fpsTimer.Tick += (s, te) =>
            {
                watch.Stop();
                float fps = counts * 1000.0f / watch.ElapsedMilliseconds;
                m_fps.Text = fps.ToString();
                counts = 0;
                watch.Reset();
                watch.Start();
            };

            im = FxTools.FxImages_safe_constructors(nextFrame.ToBitmap());
            imEl = new ImageElement(im);
            imAv = new ImageElement(nextMat.Resize(nextFrame.Width / 2, nextFrame.Height / 2));

            canvas1.AddElements(imEl);
            canvas1.AddElements(imAv);
            imAv.Position = new FxMaths.Vector.FxVector2f(nextFrame.Width, 0);

            _running = true;
            captureThread.Start();
            fpsTimer.Start();

            propertyGrid1.SelectedObject = cameraConfigs;
        }

        FxMatrixMask G;
        FxMatrixF s, m;

        private void CaptureCam()
        {

            G = new FxMatrixMask(nextMat.Width, nextMat.Height, false);
            s = new FxMatrixF(nextMat.Width, nextMat.Height, 1f);
            m = nextMat;

            /* create a circle with specific radius */
            double step = Math.PI / 50;
            var c = new FxMatrixF(nextMat.Width / 2, nextMat.Height / 2, 0f);

            while (_running)
            {

                if (nextFrame != null)
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

                var grad = nextMat.Gradient(cameraConfigs.edgeDetect);
                grad.Divide(grad.Max());

                imEl.UpdateInternalImage(grad, cameraConfigs.camFrameMap);

                var mask = (grad.Resize(grad.Width / 2, grad.Height / 2)) > 0.2f;
                //imAv.UpdateInternalImage(mask.ToFxMatrixF(), cameraConfigs.camResultMap);

                /* in all mask point add a circle */
                c.SetValue(0);
                for (int x = 0; x < mask.Width; x++)
                {
                    for (int y = 0; y < mask.Height; y++)
                    {
                        if (mask[x, y])
                        {
                            for (double t = 0; t < 2 * Math.PI; t += step)
                            {
                                int i = (int)(x + cameraConfigs.rad * Math.Cos(t));
                                int j = (int)(y + cameraConfigs.rad * Math.Sin(t));
                                if (i >= 0 && i < mask.Width && j >= 0 && j < mask.Height)
                                    c[i, j] += 1.0f;
                            }
                        }
                    }
                }
                c.Divide(c.Max());
                //c.Clamp(0.8f, 1.0f);
                mask = c > 0.9f;
                c.SetValue(0);
                for (int x = 0; x < mask.Width; x++)
                {
                    for (int y = 0; y < mask.Height; y++)
                    {
                        if (mask[x, y])
                        {
                            for (double t = 0; t < 2 * Math.PI; t += step)
                            {
                                int i = (int)(x + cameraConfigs.rad * Math.Cos(t));
                                int j = (int)(y + cameraConfigs.rad * Math.Sin(t));
                                if (i >= 0 && i < mask.Width && j >= 0 && j < mask.Height)
                                    c[i, j] += 1.0f;
                            }
                        }
                    }
                }
                c.Divide(c.Max());
                //var cSmall = c.Resize(mask.Width / 2, mask.Height / 2);

                imAv.UpdateInternalImage(c + nextMat.Resize(grad.Width / 2, grad.Height / 2), cameraConfigs.camResultMap);

                counts++;
                canvas1.ReDraw();
            }

        }

        private void CameraTesting_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (captureThread != null)
            {
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
            //Bitmap circles = new Bitmap(@"C:\Users\FxBit\Desktop\test.jpg");

            var mat = FxMatrixF.Load(circles, FxMaths.Matrix.ColorSpace.Grayscale);
            var grad = mat.Gradient(FxMatrixF.GradientMethod.Roberts);
            //canvas1.AddElements(new ImageElement(grad));

            /* create a circle with specific radius */
            int rad = 7;
            double step = Math.PI / 50;

            /* ovelap the circle to the grad */
            grad.Divide(grad.Max());
            var mask = (mat > 0.5f).MedianFilt(3, 3);
            canvas1.AddElements(new ImageElement(mask.ToFxMatrixF()));


            ///* in all mask point add a circle */
            //var c = new FxMatrixF(mask.Width, mask.Height);
            //TimeStatistics.StartClock();
            //for(int x=0; x < mask.Width; x++) {
            //    for(int y=0; y < mask.Height; y++) {
            //        if(mask[x, y]) {
            //            for(double t=0; t < 2 * Math.PI; t += step) {
            //                int i = (int)(x + rad * Math.Cos(t));
            //                int j = (int)(y + rad * Math.Sin(t));
            //                if(i >= 0 && i < mask.Width && j >= 0 && j < mask.Height)
            //                    c[i, j] += 1.0f;
            //            }
            //        }
            //    }
            //}
            //TimeStatistics.StopClock();
            //c.Divide(c.Max());
            //c.SaveCsv("test.csv");


            TimeStatistics.StartClock();
            FxMatrixF labels;
            int count;
            labels = mask.Labeling(out count);
            labels.Divide(labels.Max());
            TimeStatistics.StopClock();

            canvas1.AddElements(new ImageElement(labels, new ColorMap(ColorMapDefaults.Jet)));

            




            TimeStatistics.StartClock();
            BlobDetector bd = new BlobDetector(Emgu.CV.CvEnum.BLOB_DETECTOR_TYPE.CC);
            Bitmap bit = mask.ToBitmap();
            Image<Gray, Byte> im = new Image<Gray, Byte>(bit);
            BlobSeq blobSeq = new BlobSeq();
            BlobSeq emptyBlobSeq = new BlobSeq();
            bd.DetectNewBlob(im, blobSeq, emptyBlobSeq);
            TimeStatistics.StopClock();

            for (int i = 0; i < blobSeq.Count; i++)
            {
                MCvBlob blob = blobSeq[i];
                Console.WriteLine("Blob size: " + blob.Size);
            }

            //ImageViewer viewer = new ImageViewer();
            //viewer.Image = im;
           // viewer.Show();
        }


        #endregion











        private void CaptureVideo()
        {
            ImageElement imF = new ImageElement(nextMat);
            canvas1.AddElements(imF, false);

            ImageElement imM = new ImageElement(nextMat);
            imM.Position.Y += nextMat.Height;
            canvas1.AddElements(imM, false);

            ImageElement imS = new ImageElement(nextMat);
            imS.Position.x += nextMat.Width;
            canvas1.AddElements(imS, false);

            ImageElement imG = new ImageElement(nextMat);
            imG.Position.x += nextMat.Width;
            imG.Position.Y += nextMat.Height;
            canvas1.AddElements(imG, false);

            ImageElement imG_small = new ImageElement(nextMat);
            imG_small.Position.x += 2*nextMat.Width;
            imG_small.Position.Y += nextMat.Height;
            canvas1.AddElements(imG_small, false);

            canvas1.FitView();

            var m = nextMat.Copy();
            var s = nextMat.Copy();
            var G = nextMat != -1;
            var c = new FxMatrixF(2 * 64, 2 * 48);
            var cG = c != -1;
            var step_w = (int)Math.Ceiling(G.Width / (2 * 64.0));
            var step_h = (int)Math.Ceiling(G.Height / (2 * 48.0));
            var cG_thd = step_w * step_h / 3;

            //double step = Math.PI / 20;

            BlobTrackerAutoParam<Bgr> param = new BlobTrackerAutoParam<Bgr>();
            param.FGDetector = new FGDetector<Bgr>(Emgu.CV.CvEnum.FORGROUND_DETECTOR_TYPE.FGD);
            param.FGTrainFrames = 10;
           // param.BlobTracker = new BlobTracker(Emgu.CV.CvEnum.BLOBTRACKER_TYPE.CCMSPF);

            BlobTrackerAuto<Bgr> tracker = new BlobTrackerAuto<Bgr>(param);
            var nextFrame = capture.QueryFrame();
            int skip_count = 50;

            nextMat.Load(nextFrame.Bytes, FxMaths.Matrix.ColorSpace.Grayscale);
            FxBlobTracker fxtracker = new FxBlobTracker(nextMat);

            while (_running)
            {
                /* Load new frame */
                if (nextFrame != null)
                    nextFrame.Dispose();

                /* read next frame */
                nextFrame = capture.QueryFrame();
                if (nextFrame == null)
                    break;

                nextMat.Load(nextFrame.Bytes, FxMaths.Matrix.ColorSpace.Grayscale);

                /* process the new matrix */
                fxtracker.Process(nextMat);


                var blobs = new FxContour(fxtracker.G_small);
                var result = blobs.ToFxMatrixF(64, 48);
                var cMat = nextMat.Copy();

                foreach (FxBlob b in fxtracker.ListBlobs)
                {
                    result.DrawCircle(b.Center, b.Radius, 1f);
                    cMat.DrawCircle(b.Center * fxtracker.step_w, b.Radius * fxtracker.step_w, 1);
                }

                /* update images */
                imF.UpdateInternalImage(cMat, cameraConfigs.camFrameMap);
                //imM.UpdateInternalImage(fxtracker.m, cameraConfigs.camFrameMap);
                imS.UpdateInternalImage(fxtracker.s, cameraConfigs.camFrameMap);
                imG.UpdateInternalImage(fxtracker.FGMask.ToFxMatrixF(), cameraConfigs.camFrameMap);
                imG_small.UpdateInternalImage(fxtracker.G_small.ToFxMatrixF(), cameraConfigs.camFrameMap);



                imM.UpdateInternalImage(result, cameraConfigs.camFrameMap);
                
                /* refresh images */
                counts++;
                canvas1.ReDraw();
            }

        }


        private void toolStripButton4_Click(object sender, EventArgs e)
        {

        }

        private void toolStripButton3_Click(object sender, EventArgs e)
        {
            // init the camera capturing
            //capture = new Capture(@"C:\Dropbox\Didaktoriko\SmartCam\Video\MVI_6109.MOV");

            capture = new Capture(@"C:\Dropbox\Didaktoriko\SmartCam\Video\MVI_6109.MOV");

            //capture.SetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_AUTO_EXPOSURE, 1);
            //  capture.SetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_MODE, 1);
            //capture.SetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_FPS, 30);
            //capture.SetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_FRAME_WIDTH, 640);
            //capture.SetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_FRAME_HEIGHT, 480);
            Console.WriteLine("Width:" + capture.GetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_FRAME_WIDTH));
            Console.WriteLine("Height:" + capture.GetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_FRAME_HEIGHT));

            capture.SetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_POS_FRAMES, 900);
            nextFrame = capture.QueryGrayFrame();

            nextMat = FxMatrixF.Load(nextFrame.Bytes,
                                     nextFrame.Width,
                                     nextFrame.Height,
                                     FxMaths.Matrix.ColorSpace.Grayscale);

            // create the capture thread
            captureThread = new Thread(CaptureVideo);

            // add the timer for the fps measure
            fpsTimer = new System.Windows.Forms.Timer();
            fpsTimer.Interval = 1000;
            watch.Start();

            fpsTimer.Tick += (s, te) =>
            {
                watch.Stop();
                float fps = counts * 1000.0f / watch.ElapsedMilliseconds;
                m_fps.Text = fps.ToString();
                counts = 0;
                watch.Reset();
                watch.Start();
            };

            _running = true;
            captureThread.Start();
            fpsTimer.Start();

            propertyGrid1.SelectedObject = cameraConfigs;
        }

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
        public float rad { get; set; }

        public FxMatrixF.GradientMethod edgeDetect { get; set; }


        public CameraConfigs()
        {
            a = 0.9f;
            b = 0.3f;
            edgeDetect = FxMatrixF.GradientMethod.Sobel;
            rad = 5;
        }
    }
}
