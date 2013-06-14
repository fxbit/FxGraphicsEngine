
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

using SharpDX;
using SharpDX.Direct3D11;
using SharpDX.MediaFoundation;

using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using GraphicsEngine.Core;

namespace Tester
{
    public class MediaPlayer
    {
        private Capture capture;
        private Device dev;
        private bool isVideoStopped;
        private readonly object lockObject = new object();

        /// <summary>
        /// We need to update the resource on render target.
        /// </summary>
        private bool isDirty = false;

        /// <summary>
        /// Gets whether this media player is playing a video or audio.
        /// </summary>
        public bool IsPlaying { get; private set; }

        /// <summary>
        /// Gets or sets the background color used to display the video.
        /// </summary>
        public Color BackgroundColor { get; set; }

        public Size2 VideoSize { get; set; }

        Texture texture;
        Image<Bgr, byte> nextFrame;

        public MediaPlayer()
        {
            BackgroundColor = Color.Transparent;
            isVideoStopped = true;
        }

        public MediaPlayer(String filePath)
        {
            BackgroundColor = Color.Transparent;
            isVideoStopped = true;
            SetFile(filePath);
        }


        public virtual void Initialize(Device dev)
        {
            
            lock (lockObject)
            {
                // store internal the devices
                this.dev = dev;
            }


        }



        public void Shutdown()
        {
            lock (lockObject)
            {

            }
        }

        public virtual void OnRender(Texture2D targetBase)
        {
            lock (lockObject)
            {
                DataStream datastream;

                if (capture == null)
                    return;

                if(nextFrame!=null)
                    nextFrame.Dispose();

                // get the next frame
                nextFrame = capture.QueryFrame();

                // lock the data for reading
                System.Drawing.Bitmap bitmap = nextFrame.ToBitmap();
                System.Drawing.Imaging.BitmapData bitdata = bitmap.LockBits(new System.Drawing.Rectangle(0, 0, VideoSize.Width, VideoSize.Height),
                                                                            System.Drawing.Imaging.ImageLockMode.ReadOnly,
                                                                            System.Drawing.Imaging.PixelFormat.Format32bppRgb);
                
                // update the texture to the next data
                dev.ImmediateContext.MapSubresource(targetBase, 0, MapMode.WriteDiscard, MapFlags.None, out datastream);

                datastream.Write(bitdata.Scan0, 0, bitdata.Height*bitdata.Stride);

                dev.ImmediateContext.UnmapSubresource(targetBase, 0);

                // unlock the data
                bitmap.UnlockBits(bitdata);
                bitmap.Dispose();

                return;

                if (isVideoStopped)
                    return;

                if (!isDirty)
                    return;

                // update the texture


                isDirty = false;

            }
        }


        /// <summary>
        /// Pauses the audio/video.
        /// </summary>
        public void Pause()
        {

        }


        /// <summary>
        /// Steps forward or backward one frame.
        /// </summary>
        public void FrameStep(bool forward)
        {

        }

        /// <summary>
        /// Gets the duration of the audio/video.
        /// </summary>
        public double Duration
        {
            get
            {
                double duration = 0.0;

                return duration;
            }
        }


        /// <summary>
        /// Plays the audio/video.
        /// </summary>
        public void Play()
        {

        }


        private void StopVideo()
        {
            isVideoStopped = true;
            IsPlaying = false;
        }


        public void SetFile(string filePath)
        {
            if (capture != null)
                capture.Dispose();

            capture = new Capture(filePath);
            
            nextFrame = capture.QueryFrame();
            if (nextFrame != null)
                isDirty = true;

            this.VideoSize = new Size2((int)capture.GetCaptureProperty( Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_FRAME_WIDTH),
                                       (int)capture.GetCaptureProperty( Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_FRAME_HEIGHT));

#if false
            Image<Bgr, byte> img = capture.QueryFrame();
            img = capture.QueryFrame();
            ImageViewer viewer = new ImageViewer();
            viewer.Image = img;
            viewer.ShowDialog();
#endif
        }

        public Texture CreateTexture()
        {
            /// create the texture base on the path
            Texture newTex = new Texture();
            newTex.Path = "";
            newTex.ScaleU = 1f;
            newTex.ScaleV = 1f;
            newTex.Alpha = 1f;

            System.Drawing.Bitmap bitmap = nextFrame.ToBitmap();
            System.Drawing.Imaging.BitmapData bitdata =  bitmap.LockBits(new System.Drawing.Rectangle(0,0,VideoSize.Width,VideoSize.Height), System.Drawing.Imaging.ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format32bppRgb);
            newTex.texture2D = new SharpDX.Direct3D11.Texture2D(dev, new SharpDX.Direct3D11.Texture2DDescription()
            {
                Width = this.VideoSize.Width,
                Height = this.VideoSize.Height,
                ArraySize = 1,
                BindFlags = SharpDX.Direct3D11.BindFlags.ShaderResource,
                Usage = SharpDX.Direct3D11.ResourceUsage.Dynamic,
                CpuAccessFlags = SharpDX.Direct3D11.CpuAccessFlags.Write,
                Format = SharpDX.DXGI.Format.B8G8R8A8_UNorm,
                MipLevels = 1,
                OptionFlags = SharpDX.Direct3D11.ResourceOptionFlags.None,
                SampleDescription = new SharpDX.DXGI.SampleDescription(1, 0),
            }, new SharpDX.DataRectangle(bitdata.Scan0, bitdata.Stride));
            bitmap.UnlockBits(bitdata);

            newTex.shaderResource = new ShaderResourceView(dev, newTex.texture2D);

            // delete any old texture
            if (texture != null)
                texture.texture2D.Dispose();

            // save the new texture localy
            texture = newTex;
            return newTex;
        }
    }
}
