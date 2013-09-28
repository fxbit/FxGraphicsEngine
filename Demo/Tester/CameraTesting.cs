﻿//#define USE_BGR

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


namespace Tester
{
    public partial class CameraTesting : DockContent
    {
        Capture capture;

#if USE_BGR
        Image<Bgr, byte> nextFrame;
#else
        Image<Gray, byte> nextFrame;
#endif

        Thread captureThread;
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

        public CameraTesting()
        {
            InitializeComponent();

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
            fpsTimer.Tick += (sender, e) => {
                watch.Stop();
                float fps = counts * 1000.0f / watch.ElapsedMilliseconds;
                m_fps.Text = fps.ToString();
                counts = 0;
                watch.Reset();
                watch.Start();
            };
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
            imAv.Position = new FxMaths.Vector.FxVector2f(nextFrame.Width, 0);

            _running = true;
            captureThread.Start();
            fpsTimer.Start();
        }

        float a = 0.5f;
        float b = 0.1f;
        FxMatrixMask G;
        FxMatrixF s,m;

        private void CaptureCam()
        {

            G=new FxMatrixMask(nextMat.Width, nextMat.Height,false);
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

                var mask = nextMat > 0.5f;//((nextMat.Max() +nextMat.Min())/2);

                var diff = nextMat - m;
                s = (a + G * (b - a)) * (diff * diff - s) + s;
                m = (a + G * (b - a)) * diff + m;
                G = s > 40;

                //average[mask] = average * 0.9f + nextMat * 0.1f;

                //average[mask] = nextMat;
                //average = nextMat[mask];
                
                //  average.Multiply(0.9f);
                //  average.Add(mat * 0.1f);

                FxMatrixF r = s.Copy();
                r.Subtract(r.Min());
                r.Divide(r.Max());

                nextMat.Subtract(nextMat.Min());
                nextMat.Divide(nextMat.Max());

                for (int i = 0; i < 256; i++) {
                    for (int j = 0; j < 20;j++ )
                        nextMat[i, j] = (i % 256) / 255.0f;
                }
                    

                var cmap = new ColorMap(ColorMapDefaults.Jet);
                imEl.UpdateInternalImage(nextMat, cmap);
                imAv.UpdateInternalImage(r, cmap);

                counts++;
                canvas1.ReDraw();
            }

        }

        private void CameraTesting_FormClosing(object sender, FormClosingEventArgs e)
        {
            _running = false;
            captureThread.Abort();
            captureThread.Join();
        }
    }
}
