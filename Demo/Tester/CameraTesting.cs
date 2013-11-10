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



        //The method returns a dictionary, where the key is the label
        //and the list contains all the pixels with that label
        public Dictionary<float, LinkedList<Point>> ProcessCCL(FxMatrixMask mat, out FxMatrixF labels)
        {
            //Matrix to store pixels' labels
            labels = new FxMatrixF(mat.Width, mat.Height);

            //I particulary don't like how I store the label equality table
            //But I don't know how else can I store it
            //I use LinkedList to add and remove items faster
            Dictionary<float, LinkedList<float>> equalityTable = new Dictionary<float, LinkedList<float>>();
            //Current label
            int currentKey = 1;
            for (int y = 1; y < mat.Height; y++)
            {
                for (int x = 1; x < mat.Width; x++)
                {
                    if (mat[x, y])
                    {
                        //Minumum label of the neighbours' labels
                        int label = (int)Math.Min(labels[x - 1, y], labels[x, y - 1]);

                        //If there are no neighbours
                        if (label == 0)
                        {
                            //Create a new unique label
                            labels[x, y] = currentKey;
                            equalityTable.Add(currentKey, new LinkedList<float>());
                            equalityTable[currentKey].AddFirst(currentKey);
                            currentKey++;
                        }
                        else
                        {
                            labels[x, y] = label;
                            float west = labels[x - 1, y], north = labels[x, y - 1];
                            //A little trick:
                            //Because of those "ifs" the lowest label value
                            //will always be the first in the list
                            //but I'm afraid that because of them
                            //the running time also increases
                            if (!equalityTable[label].Contains(west))
                                if (west < equalityTable[label].First.Value)
                                    equalityTable[label].AddFirst(west);
                            if (!equalityTable[label].Contains(north))
                                if (north < equalityTable[label].First.Value)
                                    equalityTable[label].AddFirst(north);
                        }
                    }
                }
            }
            for (int y = mat.Height - 2; y >= 0; y--)
            {
                for (int x = mat.Width - 2; x >= 0; x--)
                {
                    if (mat[x, y])
                    {
                        //Minumum label of the neighbours' labels
                        int label = (int)Math.Min(labels[x + 1, y], labels[x, y + 1]);

                        //If there are no neighbours
                        if (label == 0)
                        {
                            //Create a new unique label
                            labels[x, y] = currentKey;
                            equalityTable.Add(currentKey, new LinkedList<float>());
                            equalityTable[currentKey].AddFirst(currentKey);
                            currentKey++;
                        }
                        else
                        {
                            labels[x, y] = label;
                            float west = labels[x + 1, y], north = labels[x, y + 1];
                            //A little trick:
                            //Because of those "ifs" the lowest label value
                            //will always be the first in the list
                            //but I'm afraid that because of them
                            //the running time also increases
                            if (!equalityTable[label].Contains(west))
                                if (west < equalityTable[label].First.Value)
                                    equalityTable[label].AddFirst(west);
                            if (!equalityTable[label].Contains(north))
                                if (north < equalityTable[label].First.Value)
                                    equalityTable[label].AddFirst(north);
                        }
                    }
                }
            }
            //This dictionary will be returned as the result
            //I'm not proud of using dictionary here too, I guess there 
            //is a better way to store the result
            Dictionary<float, LinkedList<Point>> result = new Dictionary<float, LinkedList<Point>>();
            //I define the variable outside the loops in order 
            //to reuse the memory address
            int cellValue;
            for (int x = 0; x < mat.Width; x++)
            {
                for (int y = 0; y < mat.Height; y++)
                {
                    cellValue = (int)labels[x, y];
                    //If the pixel is not a background
                    if (cellValue != 0)
                    {
                        //Take the minimum value from the label equality table 
                        float value = equalityTable[cellValue].First.Value;
                        //I'd like to get rid of these lines
                        if (!result.ContainsKey(value))
                            result.Add(value, new LinkedList<Point>());
                        result[value].AddLast(new Point(x, y));
                    }
                }
            }
            return result;
        }






        public FxMatrixF ProcessCCL(FxMatrixMask mask)
        {
            int Width = mask.Width;
            int Height = mask.Height;
            int win = 1;
            FxMatrixF result = new FxMatrixF(Width, Height);
            FxMatrixF resultTemp = new FxMatrixF(Width, Height);

            // set a uniq id in each position
            Parallel.For(0, Height, (y) =>
            {
                int offsetEnd = (y + 1) * Width;
                int offsetX = y * Width;
                for (int x = offsetX; x < offsetEnd; x++)
                {
                    if (mask[x])
                        result[x] = x;
                }
            });

            int count = 0;
            // try to check with neighbor
            while (true)
            {
                Boolean findChanged = false;
                Parallel.For(0, Height, (y) =>
                {
                    int iy_start = (y > win) ? y - win : 0;
                    int iy_end = (y + win + 1 < Height) ? y + win + 1 : Height;
                    for (int x = 0; x < Width; x++)
                    {
                        if (mask[x, y])
                        {
                            // find the smaller id
                            float smaller = float.MaxValue;
                            int ix_start = (x > win) ? x - win : 0;
                            int ix_end = (x + win + 1 < Width) ? x + win + 1 : Width;
                            for (int iy = iy_start; iy < iy_end; iy++)
                            {
                                for (int ix = ix_start; ix < ix_end; ix++)
                                {
                                    if (mask[ix, iy] && (smaller > result[ix, iy]))
                                        smaller = result[ix, iy];
                                }
                            }
                            if (result[x, y] != smaller)
                            {
                                findChanged = true;
                                result[x, y] = smaller;
                            }
                        }
                    }
                });

                if (!findChanged)
                    break;

                count++;
            }
            Console.WriteLine("Count:" + count.ToString());
            return result;
        }


        private bool mark(int x, int y, FxMatrixMask mask, FxMatrixF result, int count)
        {
            if (x < 0 || x >= mask.Width || y < 0 || y >= mask.Height)
                return false;

            if (mask[x, y])
            {
                result[x, y] = count;
                mask[x, y] = false;
                return true;
            }
            return false;
        }

        private void addStack(int x, int y)
        {
            stack.Push(Tuple.Create(x, y - 1));
            stack.Push(Tuple.Create(x, y + 1));

            stack.Push(Tuple.Create(x + 1, y - 1));
            stack.Push(Tuple.Create(x + 1, y));
            stack.Push(Tuple.Create(x + 1, y + 1));

            stack.Push(Tuple.Create(x - 1, y - 1));
            stack.Push(Tuple.Create(x - 1, y));
            stack.Push(Tuple.Create(x - 1, y + 1));
        }

        Stack<Tuple<int, int>> stack = new Stack<Tuple<int, int>>(1000);

        public FxMatrixF ProcessCCL2(FxMatrixMask mask)
        {
            int Width = mask.Width;
            int Height = mask.Height;
            int maskSize = Width*Height;

            FxMatrixMask remainMask = mask.Copy();

            FxMatrixF labelMap = new FxMatrixF(Width, Height);
            FxMatrixF resultTemp = new FxMatrixF(Width, Height);

            int labelCount = 0;
            for (int i = 0; i < maskSize;i++)
            {
                /* find the next start point */
                if (remainMask[i])
                {
                    int x;
                    int y = Math.DivRem(i, Width, out x);
                    remainMask[i] = false;
                    labelMap[x, y] = labelCount;

                    /* propacate the search in sub pixels */
                    addStack(x, y);

                    /* 4 cases */
                    while (stack.Count > 0)
                    {
                        Tuple<int, int> dxy = stack.Pop();

                        if (mark(dxy.Item1, dxy.Item2, remainMask, labelMap, labelCount))
                        {
                            addStack(dxy.Item1, dxy.Item2);
                        }
                    }

                    labelCount++;
                }
            }

            Console.WriteLine("LabelCount:" + labelCount.ToString());

            return labelMap;
        }

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
        }


        #endregion











        private void CaptureVideo()
        {
            ImageElement imF = new ImageElement(nextMat);
            canvas1.AddElements(imF);

            ImageElement imG = new ImageElement(nextMat);
            imG.Position.x += nextMat.Width;
            canvas1.AddElements(imG);

            var m = nextMat.Copy();
            var s = nextMat.Copy();
            var G = nextMat != -1;
            var c = new FxMatrixF(64, 48);
            var cG = c != -1;
            var step_w = (int)Math.Ceiling(G.Width / 64.0);
            var step_h = (int)Math.Ceiling(G.Height / 48.0);
            var cG_thd = step_w * step_h / 3;

            double step = Math.PI / 20;

            while (_running)
            {

                /* Load new frame */
                if (nextFrame != null)
                    nextFrame.Dispose();
                nextFrame = capture.QueryGrayFrame();
                nextMat.Load(nextFrame.Bytes, FxMaths.Matrix.ColorSpace.Grayscale);

                /* detection algorithm */
                var diff = nextMat - m;
                s = (cameraConfigs.a + G * (cameraConfigs.b - cameraConfigs.a)) * (diff * diff - s) + s;
                m = (cameraConfigs.a + G * (cameraConfigs.b - cameraConfigs.a)) * diff + m;
                G = s > 0.005f;

                /* create a resize value */
                cG.SetValueFunc((x, y) =>
                {
                    int sum = 0;
                    for (int ix = x * step_w; ix < x * step_w + step_w; ix++)
                    {
                        for (int iy = y * step_h; iy < y * step_h + step_h; iy++)
                        {
                            sum += G[ix, iy] ? 1 : 0;
                        }
                    }
                    return sum > cG_thd;
                });

                //var mask = cG.ToFxMatrixF().MedianFilt().Gradient() > 0.1f;

                //// TODO: use http://en.wikipedia.org/wiki/Connected_Component_Labeling

                /////* in all mask point add a circle */
                //c.SetValue(0);
                //for(int x = 0; x < mask.Width; x++) {
                //    for(int y = 0; y < mask.Height; y++) {
                //        if(mask[x, y]) {
                //            for(double t = 0; t < 2 * Math.PI; t += step) {
                //                int i = (int)(x + cameraConfigs.rad * Math.Cos(t));
                //                int j = (int)(y + cameraConfigs.rad * Math.Sin(t));
                //                if(i >= 0 && i < mask.Width && j >= 0 && j < mask.Height)
                //                    c[i, j] += 1.0f;
                //            }
                //        }
                //    }
                //}
                //c.Divide(c.Max());
                //var cMask = c > 0.8f;

                //TimeStatistics.StartClock();
                //FxMatrixF labels;
                //var test = ProcessCCL(cG, out labels);
                //labels.Divide(labels.Max());
                //TimeStatistics.StopClock();


                TimeStatistics.StartClock();
                FxMatrixF labels;
                labels = ProcessCCL(cG);
                labels.Divide(labels.Max());
                TimeStatistics.StopClock();

                /* update image elements */
                imF.UpdateInternalImage(cG.ToFxMatrixF(), cameraConfigs.camFrameMap);
                imG.UpdateInternalImage(labels, cameraConfigs.camFrameMap);

                /* refresh images */
                counts++;
                canvas1.ReDraw();
            }

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

            capture.SetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_POS_FRAMES, 600);
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
