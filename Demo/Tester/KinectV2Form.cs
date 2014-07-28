
#define USE_KINECT

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using System.Windows.Threading;
using System.Diagnostics;

using FxMaths;
using FxMaths.Images;
using FxMaths.GUI;
using FxMaths.Vector;
using FxMaths.GMaps;
using FxMaths.Matrix;

using GraphicsEngine;
using GraphicsEngine.UI;
using GraphicsEngine.Managers;
using GraphicsEngine.Core;
using GraphicsEngine.Core.PrimaryObjects3D;
using GraphicsEngine.Core.Shaders;

using SharpDX;
using SharpDX.Direct3D11;
using SharpDX.DXGI;

using WeifenLuo.WinFormsUI.Docking;

#if USE_KINECT
using Microsoft.Kinect;
#endif


namespace Tester
{
    public partial class KinectV2Form : DockContent
    {

        #region Variables

        Engine engine;

        /// <summary>
        /// The viewport
        /// </summary>
        GraphicsEngine.Core.Viewport RenderArea_Viewport = null;



        /// <summary>
        /// Active Kinect sensor
        /// </summary>
        private KinectSensor kinectSensor = null;

        /// <summary>
        /// Coordinate mapper to map one type of point to another
        /// </summary>
        private CoordinateMapper coordinateMapper = null;

        /// <summary>
        /// Reader for all frames
        /// </summary>
        private MultiSourceFrameReader reader = null;


        /// <summary>
        /// Intermediate storage for receiving frame data from the sensor
        /// </summary>
        private ushort[] frameData = null;

        /// <summary>
        /// Intermediate storage for frame data converted to color
        /// </summary>
        private FxMatrixF depthImageMatrix = null;
        private FxMatrixF colorImageMatrix = null;
        private FxMatrixF depthImageMatrixAve = null;

        /// <summary>
        /// Intermediate storage for the depth to color mapping
        /// </summary>
        private ColorSpacePoint[] colorPoints = null;


        /// <summary>
        /// The time of the first frame received
        /// </summary>
        private TimeSpan startTime;


        /// <summary>
        /// Next time to update FPS/frame time status
        /// </summary>
        private DateTime nextStatusUpdate = DateTime.MinValue;

        /// <summary>
        /// Number of frames since last FPS/frame time status
        /// </summary>
        private uint framesSinceUpdate = 0;

        /// <summary>
        /// Timer for FPS calculation
        /// </summary>
        private Stopwatch stopwatch = null;

        private ImageElement depthImageElement;
        private ImageElement colorImageElement;

        /// <summary>
        /// Color map of depth viewer.
        /// </summary>
        private ColorMap depthColorMap = new ColorMap(ColorMapDefaults.Jet);

        #endregion



        #region Constructor - FormClosing
        public KinectV2Form(Engine engine)
        {
            InitializeComponent();

#if false
            // save localy the graphic engine
            this.engine = engine;

            // set the first viewport
            RenderArea_Viewport = new GraphicsEngine.Core.Viewport(RenderArea.Width, RenderArea.Height, RenderArea.Handle, Format.R8G8B8A8_UNorm);
            ViewportManager.AddViewport(RenderArea_Viewport);

            // set the moving camera
            Engine.g_MoveCamera = RenderArea_Viewport.m_Camera;
#endif 

            // allocate the Matrix
            colorImageMatrix = new FxMatrixF(640, 480);
            depthImageMatrix = new FxMatrixF(640, 480);
            depthImageMatrixAve = new FxMatrixF(640, 480);

            // create a new image element
            colorImageElement = new ImageElement(colorImageMatrix);
            depthImageElement = new ImageElement(depthImageMatrix);

            canvas1.AddElement(colorImageElement, false);
            canvas1.AddElement(depthImageElement, false);
        }

        private void KinectV2Form_FormClosing(object sender, FormClosingEventArgs e)
        {
            // remove from rendering the current viewport
            ViewportManager.RemoveViewport(RenderArea_Viewport);

            if (reader != null)
            {
                // DepthFrameReder is IDisposable
                reader.Dispose();
                reader = null;
            }

            if (kinectSensor != null)
            {
                kinectSensor.Close();
                kinectSensor = null;
            }
        }
        
        #endregion



        #region Send the mouse focus to graphic engine
        private void RenderArea_MouseClick(object sender, MouseEventArgs e)
        {
            if (engine != null && Engine.isEngineRunning)
            {
                if (e.Button == MouseButtons.Right)
                    engine.refocusInput();

                Engine.g_MoveCamera = RenderArea_Viewport.m_Camera;
            }
        }

        private void RenderArea_Resize(object sender, EventArgs e)
        {
            if (engine != null)
            {
                if (RenderArea_Viewport != null)
                    RenderArea_Viewport.Resize(RenderArea.Width, RenderArea.Height);
            }
        }
        #endregion




        private void toolStripButton1_Click(object sender, EventArgs e)
        {
            // create a stopwatch for FPS calculation
            this.stopwatch = new Stopwatch();

            // for Alpha, one sensor is supported
            kinectSensor = KinectSensor.GetDefault();

            if (kinectSensor != null)
            {
                // open sensor
                kinectSensor.Open();

                var frameDescription = kinectSensor.DepthFrameSource.FrameDescription;

                // get the coordinate mapper
                coordinateMapper = kinectSensor.CoordinateMapper;


                // create a new particle depth infos
                #region Point Cloud

#if false
                List<FxVector3f> Points = new List<FxVector3f>();
                List<FxVector3f> Colors = new List<FxVector3f>();
                for (int x = 0; x < 512; x += 2)
                {
                    for (int y = 0; y < 420; y += 2)
                    {
                        for (int z = 0; z < 1; z++)
                        {
                            FxVector3f p;
                            p.x = x * 0.1f;
                            p.z = y * 0.1f;
                            p.y = (float)Math.Log(p.x * p.x * p.x + p.z * p.z * p.z - 3 * p.x - 3 * p.z);
                            Points.Add(p);
                            Colors.Add(rand.NextFxVector3f());
                        }
                    }
                }

                PointCloud pc = new PointCloud(Points, Colors);


                /// add the mesh to the engine mesh list
                Engine.g_MeshManager.AddMesh(pc);
#endif
                #endregion


                // open the reader for the depth frames
                reader = kinectSensor.OpenMultiSourceFrameReader(FrameSourceTypes.Depth | FrameSourceTypes.Color);

                // allocate space to put the pixels being received and converted
                frameData = new ushort[frameDescription.Width * frameDescription.Height];
                depthImageMatrix = new FxMatrixF(frameDescription.Width, frameDescription.Height);
                depthImageMatrixAve = depthImageMatrix.Copy();
                colorPoints = new ColorSpacePoint[frameDescription.Width * frameDescription.Height];

                reader.MultiSourceFrameArrived += reader_MultiSourceFrameArrived;
            }
            else
                MessageBox.Show("Error: failed to open kinect sensor");
        }

        void reader_MultiSourceFrameArrived(object sender, MultiSourceFrameArrivedEventArgs e)
        {
            var frames = e.FrameReference.AcquireFrame();

            DepthFrameHandling(frames);
        }

        bool oneTimeShot = false;
        /// <summary>
        /// Handles the depth frame data arriving from the sensor
        /// </summary>
        /// <param name="sender">object sending the event</param>
        /// <param name="e">event arguments</param>
        private void Reader_FrameArrived(object sender, DepthFrameArrivedEventArgs e)
        {
            DepthFrameReference frameReference = e.FrameReference;

            if (this.startTime == null)
            {
                this.startTime = frameReference.RelativeTime;
            }

            try
            {
                DepthFrame frame = frameReference.AcquireFrame();

                if (frame != null)
                {
                    // DepthFrame is IDisposable
                    using (frame)
                    {
                        FrameDescription frameDescription = frame.FrameDescription;

                        #region FPS
                        this.framesSinceUpdate++;

                        // update status unless last message is sticky for a while
                        if (DateTime.Now >= nextStatusUpdate)
                        {
                            // calcuate fps based on last frame received
                            double fps = 0.0;

                            if (stopwatch.IsRunning)
                            {
                                stopwatch.Stop();
                                fps = framesSinceUpdate / stopwatch.Elapsed.TotalSeconds;
                                stopwatch.Reset();
                            }

                            nextStatusUpdate = DateTime.Now + TimeSpan.FromSeconds(1);
                            toolStripLabel_fps.Text = fps + " " + (frameReference.RelativeTime - this.startTime).ToString() + "mS";
                        }

                        if (!stopwatch.IsRunning)
                        {
                            framesSinceUpdate = 0;
                            stopwatch.Start();
                        } 
                        #endregion

                        // verify data and write the new depth frame data to the display bitmap
                        if (((frameDescription.Width * frameDescription.Height) == frameData.Length))
                        {
                            // Copy the pixel data from the image to a temporary array
                            frame.CopyFrameDataToArray(frameData);

                            coordinateMapper.MapDepthFrameToColorSpace(frameData, colorPoints);

                            // Get the min and max reliable depth for the current frame
                            ushort minDepth = frame.DepthMinReliableDistance;
                            ushort maxDepth = frame.DepthMaxReliableDistance;
                            float imaxDepth = 1.0f / maxDepth;
                            for (int i = 0; i < this.frameData.Length; ++i)
                            {
                                // Get the depth for this pixel
                                ushort depth = this.frameData[i];

                                // To convert to a byte, we're discarding the most-significant
                                // rather than least-significant bits.
                                // We're preserving detail, although the intensity will "wrap."
                                // Values outside the reliable depth range are mapped to 0 (black).
                                //pixels[i] = 1.0f - ((depth >= minDepth && depth <= maxDepth) ? depth : 0) * imaxDepth;
                                depthImageMatrix[i] = 1.0f - depth * imaxDepth;
                            }

                            if (!oneTimeShot)
                            {
                                oneTimeShot = true;
                                ColorMap cm = new ColorMap(ColorMapDefaults.DeepBlue);
                                List<FxVector3f> Points = new List<FxVector3f>();
                                List<FxVector3f> Colors = new List<FxVector3f>();
                                for (int x = 0; x < frameDescription.Width; x++)
                                {
                                    for (int y = 0; y < frameDescription.Height; y++)
                                    {
                                        FxVector3f p;
                                        p.x = x * 0.08f;
                                        p.z = y * 0.08f;
                                        p.y = depthImageMatrix[x,y]*5.0f;
                                        Points.Add(p);

#if false
                                        // calculate index into depth array
                                        int depthIndex = (x * frameDescription.Width) + y;

                                        // retrieve the depth to color mapping for the current depth pixel
                                        ColorSpacePoint colorPoint = this.colorPoints[depthIndex];

                                        // make sure the depth pixel maps to a valid point in color space
                                        int colorX = (int)Math.Floor(colorPoint.X + 0.5);
                                        int colorY = (int)Math.Floor(colorPoint.Y + 0.5);

                                        // calculate index into color array
                                        int colorIndex = ((colorY * colorWidth) + colorX) * this.bytesPerPixel;


#else
                                        byte b = (byte)(depthImageMatrix[x, y]*256);
                                        Colors.Add(new FxVector3f(cm[b,0]/256.0f, cm[b,1]/256.0f, cm[b,2]/256.0f));
#endif
                                    }
                                }

                                PointCloud pc = new PointCloud(Points, Colors);

                                /// add the mesh to the engine mesh list
                                Engine.g_MeshManager.AddMesh(pc);
                            }
                        }
                    }
                }
            }
            catch (Exception)
            {
                // ignore if the frame is no longer available
            }
        }

        private void toolStripButton2_Click(object sender, EventArgs e)
        {
            MultiSourceFrame frame = reader.AcquireLatestFrame();

            try
            {
                ColorFrame colorFrame = frame.ColorFrameReference.AcquireFrame();


            }
            catch (Exception) { }


            DepthFrameHandling(frame);
        }

        private void DepthFrameHandling(MultiSourceFrame frame)
        {
            try
            {
                DepthFrame depthFrame = frame.DepthFrameReference.AcquireFrame();
                DepthFrameReference frameReference = frame.DepthFrameReference;
                if (depthFrame != null)
                {
                    // DepthFrame is IDisposable
                    using (depthFrame)
                    {
                        FrameDescription frameDescription = depthFrame.FrameDescription;

                        #region FPS
                        this.framesSinceUpdate++;

                        // update status unless last message is sticky for a while
                        if (DateTime.Now >= nextStatusUpdate)
                        {
                            // calcuate fps based on last frame received
                            double fps = 0.0;

                            if (stopwatch.IsRunning)
                            {
                                stopwatch.Stop();
                                fps = framesSinceUpdate / stopwatch.Elapsed.TotalSeconds;
                                stopwatch.Reset();
                            }

                            nextStatusUpdate = DateTime.Now + TimeSpan.FromSeconds(1);
                            toolStripLabel_fps.Text = fps + " " + (frameReference.RelativeTime - this.startTime).ToString() + "mS";
                        }

                        if (!stopwatch.IsRunning)
                        {
                            framesSinceUpdate = 0;
                            stopwatch.Start();
                        }
                        #endregion

                        // verify data and write the new depth frame data to the display bitmap
                        if (((frameDescription.Width * frameDescription.Height) == frameData.Length))
                        {
                            // Copy the pixel data from the image to a temporary array
                            depthFrame.CopyFrameDataToArray(frameData);

                            coordinateMapper.MapDepthFrameToColorSpace(frameData, colorPoints);

                            // Get the min and max reliable depth for the current frame
                            ushort minDepth = depthFrame.DepthMinReliableDistance;
                            ushort maxDepth = depthFrame.DepthMaxReliableDistance;
                            float imaxDepth = 1.0f / maxDepth;
                            for (int i = 0; i < this.frameData.Length; ++i)
                            {
                                // Get the depth for this pixel
                                ushort depth = this.frameData[i];

                                // To convert to a byte, we're discarding the most-significant
                                // rather than least-significant bits.
                                // We're preserving detail, although the intensity will "wrap."
                                // Values outside the reliable depth range are mapped to 0 (black).
                                //pixels[i] = 1.0f - ((depth >= minDepth && depth <= maxDepth) ? depth : 0) * imaxDepth;
                                depthImageMatrix[i] = 1.0f - depth * imaxDepth;
                            }

                            depthImageMatrixAve.Multiply(0.85f);
                            depthImageMatrixAve += 0.15f*depthImageMatrix;

                            // Updating viewports...
                            depthImageElement.UpdateInternalImage(depthImageMatrix, depthColorMap, true);
                            colorImageElement.UpdateInternalImage(depthImageMatrixAve, depthColorMap, true);
                            canvas1.ReDraw();

                        }
                    }
                }
            }
            catch (Exception) { }
        }

        private void toolStripButton3_Click(object sender, EventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            if (sfd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                ColorMap map = new ColorMap(ColorMapDefaults.Gray);
                lock (depthImageMatrix)
                {
                    depthImageMatrix.SaveImage(sfd.FileName + "_depth.jpg", map);
                    depthImageMatrix.SaveCsv(sfd.FileName + "_depth.csv");
                }


                lock(depthImageMatrixAve)
                {
                    depthImageMatrixAve.SaveImage(sfd.FileName + "_ave_depth.jpg", map);
                    depthImageMatrixAve.SaveCsv(sfd.FileName + "_ave_depth.csv");
                }
            }

        }
















        #region Ellipse Code


        FxMatrixF imMat = null;
        ImageElement ieEllipseImage = null;
        GeometryPlotElement gpeEllipseImage = null;
        FxMaths.Geometry.Rectangle rect = null;
        FxMatrixF results = null; // Results of centers 


        private void toolStripButton_EllipseOpenImage_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            if(ofd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                imMat = FxMatrixF.LoadCsv(ofd.FileName);
                ieEllipseImage = new ImageElement(imMat, new ColorMap(ColorMapDefaults.Jet));
                ieEllipseImage.lockMoving = true;
                canvas_ellipse.AddElement(ieEllipseImage);
            }
        }



        private void canvas_ellipse_OnCanvasMouseClick(object sender, CanvasMouseClickEventArgs e)
        {
            Console.WriteLine("CanvasHit " + e.hitPoint.ToString() + "  " + e.insidePoint + " ");
            if (e.hitElement != null)
                Console.Write("Hit:" + e.hitElement.ToString());

            // check if we are sub region selection
            if (toolStripButton_eclipse_SelectSubRegion.Checked)
            {
                // check if we are hitting the image
                if (e.hitElement == ieEllipseImage)
                {
                    // init the rectangle
                    if (rect == null)
                    {
                        rect = new FxMaths.Geometry.Rectangle(e.hitPoint, e.hitPoint);
                        gpeEllipseImage.AddGeometry(rect);
                    }
                    else
                    {
                        rect.EndPoint = e.hitPoint;
                        toolStripButton_eclipse_SelectSubRegion.Checked = false;
                    }
                }
            }
        }




        private void toolStripButton_eclipse_SelectSubRegion_Click(object sender, EventArgs e)
        {
            if (toolStripButton_eclipse_SelectSubRegion.Checked)
            {
                toolStripButton_eclipse_SelectSubRegion.Checked = false;

                // clean old geometry element
                if(gpeEllipseImage!=null)
                    gpeEllipseImage.ClearGeometry(true);

                if (rect != null)
                {
                    rect.Dispose();
                    rect = null;
                }
            }
            else
            {

                toolStripButton_eclipse_SelectSubRegion.Checked = true;

                // init geometry plot
                if (gpeEllipseImage == null)
                {
                    gpeEllipseImage = new GeometryPlotElement();
                    canvas_ellipse.AddElement(gpeEllipseImage);
                    gpeEllipseImage.Position = ieEllipseImage._Position;
                }
                else
                    gpeEllipseImage.ClearGeometry(true);


                if (rect != null)
                {
                    rect.Dispose();
                    rect = null;
                }
            }
        }




        private void toolStripButton_ellipse_extract_Click(object sender, EventArgs e)
        {
            // extract the rectangle that contain 
            if ((imMat as object != null) &&
                (rect != null))
            {
                int numLabels;

                // get the sub matrix
                var subMat = imMat.GetSubMatrix(rect.StartPoint, rect.EndPoint) as FxMatrixF;

                // make binary the image based on the start and end point of rectangle
                float t = subMat[0, 0];
                if (t > subMat[subMat.Width - 1, subMat.Height - 1])
                    t = subMat[subMat.Width - 1, subMat.Height - 1];
                var binMat = subMat < t - 0.02f;
                
                // find the labels of the image
                var labels = binMat.Labeling(out numLabels);

                var imSub = new ImageElement(binMat.ToFxMatrixF(), new ColorMap(ColorMapDefaults.Jet));
                imSub._Position.x = ieEllipseImage.Size.x + ieEllipseImage.Position.x;
                imSub.lockMoving = true;
                canvas_ellipse.AddElement(imSub, false, false);

                var imSub2 = new ImageElement(labels, new ColorMap(ColorMapDefaults.Jet));
                imSub2._Position.x = ieEllipseImage.Size.x + ieEllipseImage.Position.x;
                imSub2._Position.y = imSub.Size.y + imSub.Position.y;
                imSub2.lockMoving = true;
                canvas_ellipse.AddElement(imSub2, false, false);

                WriteLine("Num Labels : " + numLabels.ToString());

                var contours = new FxContour(binMat, 10, 300);

                WriteLine("Num Contours : " + contours.NumChains);

                results = new FxMatrixF(3, contours.NumChains);

                int i=0;
                foreach (var x in contours.ChainList)
                {
                    float delta = 1;
                    // draw the rectanges in the sub image
                    FxVector2f start = x.RectStart + imSub2._Position + new FxVector2f(1, 1);
                    FxVector2f end = start + x.RectSize;
                    FxMaths.Geometry.Rectangle r = new FxMaths.Geometry.Rectangle(start, end);
                    gpeEllipseImage.AddGeometry(r, false);


                    // draw the rectanges in main image
                    start = x.RectStart + rect.StartPoint + new FxVector2f(-delta, -delta);
                    end = start + x.RectSize + new FxVector2f(2 * delta, 2 * delta);
                    r = new FxMaths.Geometry.Rectangle(start, end);
                    gpeEllipseImage.AddGeometry(r, false);


                    // draw the centroids 
                    FxVector2f cen = x.GetCentroid() + rect.StartPoint;
                    var l = new FxMaths.Geometry.Line(cen - new FxVector2f(0, 2), cen + new FxVector2f(0, 2));
                    l.LineWidth = 0.5f;
                    gpeEllipseImage.AddGeometry(l, false);
                    l = new FxMaths.Geometry.Line(cen - new FxVector2f(2, 0), cen + new FxVector2f(2, 0));
                    l.LineWidth = 0.5f;
                    gpeEllipseImage.AddGeometry(l, false);


                    // calculate the depth
                    float[] depth = new float[4];
                    FxVector2f pos = x.RectStart + rect.StartPoint;
                    depth[0] = imMat[pos.x - delta, pos.y - delta];
                    depth[1] = imMat[pos.x + x.RectSize.x + delta, pos.y - delta];
                    depth[2] = imMat[pos.x - delta, pos.y + x.RectSize.y + delta];
                    depth[3] = imMat[pos.x + x.RectSize.x + delta, pos.y + x.RectSize.y + delta];
                    results[2, i] = (depth[0] + depth[1] + depth[2] + depth[3]) / 4.0f;


                    // save centroid
                    results[0, i] = cen.x;
                    results[1, i] = cen.y;

                    // print the centroid
                    WriteLine("[" + i.ToString() + "] Depth:" + results[2, i].ToString() + "  Pixels:" + x.Count + " Centroid: " + cen.ToString());
                    i++;


                    // show the vector of one blob
                    if (i == 1)
                    {
                        FxVectorF vec_i = new FxVectorF(x.Count);
                        FxVectorF vec_r = new FxVectorF(x.Count);
                        vec_i[0] = x[0].i;
                        vec_r[0] = x[0].r;
                        for (int j = 1; i < x.Count; j++)
                        {
                            vec_i[j] = vec_i[j - 1] + x[j].i;
                            vec_r[j] = vec_r[j - 1] + x[j].r;
                        }


                    }
                }

                canvas_ellipse.ReDraw();

                results.SaveCsv("ellipseResults.csv");

            }
        }




        #endregion







        void WriteLine(String str)
        {
            // write to the form
            //Console_Text.Text += str + "\n";

            // write to the console
            //Console.WriteLine(str);

            Tester.TesterForm.UIConsole.WriteLine(str);
        }


















        private void ellipseDetectionToolStripMenuItem_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            if(ofd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {

                // load image
                imMat = FxMatrixF.Load(ofd.FileName);
                ieEllipseImage = new ImageElement(imMat, new ColorMap(ColorMapDefaults.Jet));
                ieEllipseImage.lockMoving = true;
                canvas_ellipse.AddElement(ieEllipseImage);

                // add plot element 
                gpeEllipseImage = new GeometryPlotElement();
                canvas_ellipse.AddElement(gpeEllipseImage);


                var contours = new FxContour(imMat<0.5f);

                WriteLine("Num Contours : " + contours.NumChains);

                int i = 0;
                foreach (var x in contours.ChainList)
                {
                    // draw the rectanges in main image
                    FxVector2f start = x.RectStart ;
                    FxVector2f end = start + x.RectSize;
                    FxMaths.Geometry.Rectangle r = new FxMaths.Geometry.Rectangle(start, end);
                    gpeEllipseImage.AddGeometry(r, false);


                    // draw the centroids 
                    FxVector2f cen = x.GetCentroid();
                    var l = new FxMaths.Geometry.Line(cen - new FxVector2f(0, 2), cen + new FxVector2f(0, 2));
                    l.LineWidth = 0.5f;
                    gpeEllipseImage.AddGeometry(l, false);
                    l = new FxMaths.Geometry.Line(cen - new FxVector2f(2, 0), cen + new FxVector2f(2, 0));
                    l.LineWidth = 0.5f;
                    gpeEllipseImage.AddGeometry(l, false);


                    i++;
                    // show the vector of one blob
                    if (i == 1)
                    {
                        FxVectorF vec_i = new FxVectorF(x.Count);
                        FxVectorF vec_r = new FxVectorF(x.Count);
                        vec_i[0] = x[0].i;
                        vec_r[0] = x[0].r;
                        for (int j = 1; j < x.Count; j++)
                        {
                            vec_i[j] = vec_i[j - 1] + x[j].i;
                            vec_r[j] = vec_r[j - 1] + x[j].r;
                        }
                    }
                }

                canvas_ellipse.ReDraw();
            }
        }
    }
}
