using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;
#if USE_KINECT
using Microsoft.Kinect;
#endif
using FxMaths.Matrix;
using FxMaths.GUI;
using System.IO;
using System.Windows.Threading;
using FxMaths.Images;

namespace Tester
{
    public partial class KinectForm : DockContent
    {

        public KinectForm()
        {
            InitializeComponent();
        }


#if USE_KINECT
        #region Internal Variables
        
        
        /// <summary>
        /// Active Kinect sensor
        /// </summary>
        private KinectSensor sensor;


        /// <summary>
        /// The resolution of the depth image to be processed.
        /// </summary>
        private const DepthImageFormat DepthFormat = DepthImageFormat.Resolution640x480Fps30;

        /// <summary>
        /// Format of color frame to use - in this basic sample we are limited to use the standard 640x480 
        /// resolution Rgb at 30fps, identical to the depth resolution.
        /// </summary>
        private const ColorImageFormat ColorFormat = ColorImageFormat.RgbResolution640x480Fps30;


        /// <summary>
        /// Image Width of depth frame
        /// </summary>
        private int depthWidth = 0;

        /// <summary>
        /// Image height of depth frame
        /// </summary>
        private int depthHeight = 0;

        /// <summary>
        /// Image width of color frame
        /// </summary>
        private int colorWidth = 0;

        /// <summary>
        /// Image height of color frame
        /// </summary>
        private int colorHeight = 0;

        /// <summary>
        /// The sensor depth frame data length
        /// </summary>
        private int frameDataLength;

        /// <summary>
        /// The count of the depth frames to be processed
        /// </summary>
        private bool processingFrame = false;

        /// <summary>
        /// Intermediate storage for the color data received from the camera in 32bit color
        /// </summary>
        private byte[] colorImagePixels;

        /// <summary>
        /// Image in Matrix Form.
        /// </summary>
        private FxMatrixF colorImageMatrix;

        /// <summary>
        /// Image viewer.
        /// </summary>
        private ImageElement colorImageElement;

        /// <summary>
        /// Intermediate storage for the extended depth data received from the camera in the current frame
        /// </summary>
        private DepthImagePixel[] depthImagePixels;


        private DepthImagePoint[] depthImagePoints;

        /// <summary>
        /// Mapping of depth pixels into color image
        /// </summary>
        private ColorImagePoint[] colorCoordinates;

        /// <summary>
        /// Image in Matrix Form.
        /// </summary>
        private FxMatrixF depthImageMatrix;
        private FxMatrixF depthImageMatrixAve;

        /// <summary>
        /// Depth viewer.
        /// </summary>
        private ImageElement depthImageElement;

        /// <summary>
        /// Color map of depth viewer.
        /// </summary>
        private ColorMap depthColorMap = new ColorMap(ColorMapDefaults.Jet);

        /// <summary>
        /// Execute dispatcher
        /// </summary>
        private Dispatcher dispatcher;

        #region Vars for FPS

        /// <summary>
        /// The timer to calculate FPS
        /// </summary>
        private DispatcherTimer fpsTimer;

        /// <summary>
        /// Timer stamp of last computation of FPS
        /// </summary>
        private DateTime lastFPSTimestamp;

        /// <summary>
        /// The count of the frames processed in the FPS interval
        /// </summary>
        private int processedFrameCount;

        /// <summary>
        /// The tracking error count
        /// </summary>
        private int trackingErrorCount;
        
        #endregion


        /// <summary>
        /// The coordinate mapper to convert between depth and color frames of reference
        /// </summary>
        private CoordinateMapper mapper;

        #endregion






        #region Constructor and loader


        private void KinectForm_Load(object sender, EventArgs e)
        {
            // Look through all sensors and start the first connected one.
            // This requires that a Kinect is connected at the time of app startup.
            // To make your app robust against plug/unplug, 
            // it is recommended to use KinectSensorChooser provided in Microsoft.Kinect.Toolkit
            foreach(var potentialSensor in KinectSensor.KinectSensors) {
                if(potentialSensor.Status == KinectStatus.Connected) {
                    this.sensor = potentialSensor;
                    break;
                }
            }

            if(null == this.sensor) {
                this.Text = "No kinnect Ready";
                return;
            }


            Size depthImageSize = GetImageSize(DepthFormat);
            depthWidth = (int)depthImageSize.Width;
            depthHeight = (int)depthImageSize.Height;

            Size colorImageSize = GetImageSize(ColorFormat);
            colorWidth = (int)colorImageSize.Width;
            colorHeight = (int)colorImageSize.Height;


            // Turn on the depth and color streams to receive frames
            sensor.DepthStream.Enable(DepthFormat);
            sensor.ColorStream.Enable(ColorFormat);
            sensor.DepthStream.Range = DepthRange.Near;

            frameDataLength = sensor.DepthStream.FramePixelDataLength;

            sensor.AllFramesReady += sensor_AllFramesReady;


            int depthImageArraySize = depthWidth * depthHeight;
            int colorImageArraySize = colorWidth * colorHeight * sizeof(int);

            // Create local color pixels buffer
            colorImagePixels = new byte[colorImageArraySize];

            // Create local depth pixels buffer
            depthImagePixels = new DepthImagePixel[depthImageArraySize];

            // Allocate the depth-color mapping points
            colorCoordinates = new ColorImagePoint[depthImageArraySize];

            // allocate the Matrix
            colorImageMatrix = new FxMatrixF(colorWidth, colorHeight);
            depthImageMatrix = new FxMatrixF(depthWidth, depthHeight);
            depthImageMatrixAve = new FxMatrixF(depthWidth, depthHeight);

            // create a new image element
            colorImageElement = new ImageElement(colorImageMatrix);
            depthImageElement = new ImageElement(depthImageMatrix);
            depthImageElement.MouseClickEvent += depthImageElement_MouseClickEvent;
            canvas1.AddElements(colorImageElement,false);
            canvas1.AddElements(depthImageElement, false);
            depthImageElement.Position = new FxMaths.Vector.FxVector2f(colorWidth, 0);

            G = new FxMatrixMask(depthWidth, depthHeight, false);
            s = new FxMatrixF(depthWidth, depthHeight, 1f);
            m = new FxMatrixF(depthWidth, depthHeight, 1f);

            // init the dispatcher class
            dispatcher = Dispatcher.CurrentDispatcher;


            // Start the sensor!
            try {
                this.sensor.Start();
            } catch(IOException ex) {
                // Device is in use
                this.sensor = null;
                this.Text = ex.Message;

                return;
            } catch(InvalidOperationException ex) {
                // Device is not valid, not supported or hardware feature unavailable
                this.sensor = null;
                this.Text = ex.Message;

                return;
            }


            // Initialize and start the FPS timer
            this.fpsTimer = new DispatcherTimer();
            this.fpsTimer.Tick += new EventHandler(this.FpsTimerTick);
            this.fpsTimer.Interval = new TimeSpan(0, 0, 1);

            this.fpsTimer.Start();

            this.lastFPSTimestamp = DateTime.UtcNow;
        }



        void depthImageElement_MouseClickEvent(CanvasElements m, FxMaths.Vector.FxVector2f location)
        {
            TesterForm.UIConsole.WriteLine("Depth:" + depthImageMatrix[location.x, location.y].ToString());
            // Console.WriteLine("Depth:" + depthImageMatrix[location.x, location.y].ToString());
        }

        #endregion



        #region Utils


        /// <summary>
        /// Get the depth image size from the input depth image format.
        /// </summary>
        /// <param name="imageFormat">The depth image format.</param>
        /// <returns>The widht and height of the input depth image format.</returns>
        private static Size GetImageSize(DepthImageFormat imageFormat)
        {
            switch(imageFormat) {
                case DepthImageFormat.Resolution320x240Fps30:
                    return new Size(320, 240);

                case DepthImageFormat.Resolution640x480Fps30:
                    return new Size(640, 480);

                case DepthImageFormat.Resolution80x60Fps30:
                    return new Size(80, 60);
            }

            throw new ArgumentOutOfRangeException("imageFormat");
        }


        /// <summary>
        /// Get the color image size from the input color image format.
        /// </summary>
        /// <param name="imageFormat">The color image format.</param>
        /// <returns>The width and height of the input color image format.</returns>
        private static Size GetImageSize(ColorImageFormat imageFormat)
        {
            switch(imageFormat) {
                case ColorImageFormat.RgbResolution640x480Fps30:
                    return new Size(640, 480);

                case ColorImageFormat.RgbResolution1280x960Fps12:
                    return new Size(1280, 960);

                case ColorImageFormat.InfraredResolution640x480Fps30:
                    return new Size(640, 480);

                case ColorImageFormat.RawBayerResolution1280x960Fps12:
                    return new Size(1280, 960);

                case ColorImageFormat.RawBayerResolution640x480Fps30:
                    return new Size(640, 480);

                case ColorImageFormat.RawYuvResolution640x480Fps15:
                    return new Size(640, 480);

                case ColorImageFormat.YuvResolution640x480Fps15:
                    return new Size(640, 480);
            }

            throw new ArgumentOutOfRangeException("imageFormat");
        }

        #endregion




        #region New Image handling

        /// <summary>
        /// Ebvent handler for Kinnect sensor's 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        void sensor_AllFramesReady(object sender, AllFramesReadyEventArgs e)
        {
            // Here we will drop a frame if we are still processing the last one
            if(!this.processingFrame) {

                // Mark that one frame will be processed
                this.processingFrame = true;

                using(ColorImageFrame colorFrame = e.OpenColorImageFrame()) {
                    if(null != colorFrame) {
                        // Copy color pixels from the image to a buffer
                        colorFrame.CopyPixelDataTo(colorImagePixels);
                        colorImageMatrix.Load(colorImagePixels, FxMaths.Matrix.ColorSpace.Grayscale);

                    }
                }

                using(DepthImageFrame depthFrame = e.OpenDepthImageFrame()) {
                    if(depthFrame != null) {
                        // Copy the depth pixel data from the image to a buffer
                        depthFrame.CopyDepthImagePixelDataTo(depthImagePixels);
                        for(int i=0; i < depthImagePixels.Length; i++) {
                            if(depthImagePixels[i].IsKnownDepth)
                                depthImageMatrix[i] = depthImagePixels[i].Depth;
                            else
                                depthImageMatrix[i] = 0;
                        }
                        
                    }
                } 

                dispatcher.BeginInvoke(DispatcherPriority.Background, (Action)(() => this.ProcessMatrix()));
            }

        }
        
        #endregion

        float a = 0.05f;
        float b = 0.1f;
        FxMatrixMask G;
        FxMatrixF s,m;

        private void ProcessMatrix()
        {

            try {
                // proccessing ...
                depthImageMatrixAve = a * depthImageMatrix + (1 - a) * depthImageMatrixAve;
                var depth = depthImageMatrixAve / depthImageMatrixAve.Max();
                //depthImageMatrix.Divide(depthImageMatrix.Max());
                //depthImageMatrix.MultiplyPointwise(colorImageMatrix);



                if(null == mapper) {
                    // Create a coordinate mapper
                    mapper = new CoordinateMapper(this.sensor);

                }

                //mapper.MapColorFrameToDepthFrame(ColorFormat, DepthFormat, depthImagePixels, depthImagePoints);
                mapper.MapDepthFrameToColorFrame(DepthFormat, depthImagePixels, ColorFormat, colorCoordinates);
                for(int i=0; i < depthWidth * depthHeight; i++) {
                    depth[i] *= colorImageMatrix[colorCoordinates[i].X, colorCoordinates[i].Y];
                }


              //  m = a * depthImageMatrix + (1 - a) * m;

                // Updating viewports...
                colorImageElement.UpdateInternalImage(colorImageMatrix);
                depthImageElement.UpdateInternalImage(depth, depthColorMap, true);
                canvas1.ReDraw();

                // The input frame was processed successfully, increase the processed frame count
                ++this.processedFrameCount;
            } catch(InvalidOperationException ex) {
                Console.WriteLine(ex.Message);
            } finally {
                this.processingFrame = false;
            }


        }


        #region FpsRelative

        /// <summary>
        /// Handler for FPS timer tick
        /// </summary>
        /// <param name="sender">Object sending the event</param>
        /// <param name="e">Event arguments</param>
        private void FpsTimerTick(object sender, EventArgs e)
        {
            // Calculate time span from last calculation of FPS
            double intervalSeconds = (DateTime.UtcNow - this.lastFPSTimestamp).TotalSeconds;

            // Calculate and show fps on status bar
            this.labelFPS.Text = ((double)this.processedFrameCount / intervalSeconds).ToString();

            // Reset frame counter
            this.processedFrameCount = 0;
            this.lastFPSTimestamp = DateTime.UtcNow;
        }

        /// <summary>
        /// Reset FPS timer and counter
        /// </summary>
        private void ResetFps()
        {
            // Restart fps timer
            if(null != this.fpsTimer) {
                this.fpsTimer.Stop();
                this.fpsTimer.Start();
            }

            // Reset frame counter
            this.processedFrameCount = 0;
            this.lastFPSTimestamp = DateTime.UtcNow;
        }
        
        #endregion




        private void button_StartKinect_Click(object sender, EventArgs e)
        {

        }


        private void button_SaveImage_Click(object sender, EventArgs e)
        {
            sensor.Stop();
            SaveFileDialog sfd = new SaveFileDialog();
            if(sfd.ShowDialog()== System.Windows.Forms.DialogResult.OK){
                ColorMap map = new ColorMap(ColorMapDefaults.Gray);
                depthImageMatrix.SaveImage(sfd.FileName + "_depth.jpg", map);
                colorImageMatrix.SaveImage(sfd.FileName + "_color.jpg", map);

                depthImageMatrix.SaveCsv(sfd.FileName + "_depth.csv");
                depthImageMatrixAve.SaveCsv(sfd.FileName + "_ave_depth.csv");
            }
            sensor.Start();
        }

        int saveCount = 0;
        private void toolStripButton1_Click(object sender, EventArgs e)
        {
            String FileName = "depth_" + saveCount.ToString() + ".csv";
            saveCount++;
            depthImageMatrixAve.SaveCsv(FileName);
        }
#else

        private void KinectForm_Load(object sender, EventArgs e)
        {

        }

        private void button_StartKinect_Click(object sender, EventArgs e)
        {

        }


        private void button_SaveImage_Click(object sender, EventArgs e)
        {
        }

        private void toolStripButton1_Click(object sender, EventArgs e)
        {
        }
#endif
    }
}

