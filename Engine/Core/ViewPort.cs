using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;


using SharpDX.Direct3D11;
using SharpDX.DXGI;
using SharpDX;
using GraphicsEngine.Core;

namespace GraphicsEngine.Core {
    public class Viewport {


        #region Public Variables

        /// <summary>
        /// RenderTarget: the variable that set the render chain 
        /// for our scene 
        /// </summary>
        public RenderTargetView m_RenderTarget;

        /// <summary>
        /// Store the buffer of the rendering
        /// </summary>
        public SwapChain m_SwapChain;

        /// <summary>
        /// Store the depth of the viewport
        /// </summary>
        public DepthStencilView m_DepthStencilView;

        /// <summary>
        /// The Viewport Camera
        /// </summary>
        public Camera m_Camera;

        /// <summary>
        /// Setup the camera viewport
        /// </summary>
        public CamViewport m_Viewport;

        /// <summary>
        /// Rasterization state description
        /// </summary>
        public RasterizerStateDescription m_RasterizerStateDesc;

        /// <summary>
        /// Rasterization state
        /// </summary>
        public RasterizerState m_RasterizerState;

        /// <summary>
        /// The buffer that store the depth of the Camera
        /// </summary>
        public Texture2D m_DepthStencil;

        /// <summary>
        /// The buffer that store the render screen of the Camera
        /// </summary>
        public Texture2D m_RenderStencil;

        #endregion


        #region Private Variables

        /// <summary>
        /// The description of the Sampling for the specific viewport
        /// </summary>
        private SampleDescription m_SampleDesc;

        /// <summary>
        /// The description of the Chain for the specific viewport
        /// </summary>
        private SwapChainDescription m_SwapDesc;

        /// <summary>
        /// Descript the mode of the viewport
        /// </summary>
        private ModeDescription m_ModeDesc;

        /// <summary>
        /// Set the Handle of the Render control
        /// </summary>
        private IntPtr m_RenderControl;
        #endregion

        private Format backFormat;
        public Viewport(int Width, int Height, IntPtr RenderControl,Format backFormat)
        {

            /// Bind the render control
            this.m_RenderControl = RenderControl;

            /// 
            this.backFormat = backFormat;
            SetView(Width, Height);

            ///////////////////////////////////////////////////////////////////////// Set the Camera

            /// Set up the camera
            m_Camera = new Camera(Width, Height);
            m_Camera.SetViewParams(
                new Vector3(0.0f, 1.0f, -10.0f),
                new Vector3(0.0f, 1.0f, 0.0f));

            ///////////////////////////////////////////////////////////////////////// Set the Viewport

            /// Setup the camera viewport
            m_Viewport = new CamViewport();

            /// No offset
            m_Viewport.X = 0;
            m_Viewport.Y = 0;

            /// Size is equal to the size of the form
            m_Viewport.Width = Width;
            m_Viewport.Height = Height;
            m_Viewport.MinZ = 0.0f;
            m_Viewport.MaxZ = 1.0f;

            ///////////////////////////////////////////////////////////////////////// Set the Rasterizer state

            m_RasterizerStateDesc = new RasterizerStateDescription();
            m_RasterizerStateDesc.CullMode = Settings.CullMode;
            m_RasterizerStateDesc.FillMode = Settings.FillMode;

            m_RasterizerState = new RasterizerState(Engine.g_device, m_RasterizerStateDesc);

        }

        private void SetView(int Width, int Height)
        {
            ///////////////////////////////////////////////////////////////////////// Set the ModeDescription
            /// Create a description of the display mode
            m_ModeDesc = new ModeDescription();

            /// Standard 32-bit RGBA
            m_ModeDesc.Format = backFormat;


            /// Refresh rate of 60Hz (60 / 1 = 60)
            m_ModeDesc.RefreshRate = new Rational(Settings.FrameRate, 1);

            /// Default
            m_ModeDesc.Scaling = DisplayModeScaling.Centered;
            m_ModeDesc.ScanlineOrdering =  DisplayModeScanlineOrder.Progressive;

            /// ClientSize is the size of the
            /// form without the title and borders
            m_ModeDesc.Width = Width;
            m_ModeDesc.Height = Height;

            ///////////////////////////////////////////////////////////////////////// Set the SampleDescription

            /// Create a description of the sampling 
            /// for multisampling or antialiasing
            m_SampleDesc = new SampleDescription();

            /// No multisampling
            m_SampleDesc.Count = 1;
            m_SampleDesc.Quality = 0;

            ///////////////////////////////////////////////////////////////////////// Set the SampleDescription

            /// Create a description of the swap 
            /// chain or front and back buffers
            m_SwapDesc = new SwapChainDescription();

            /// link the ModeDescription
            m_SwapDesc.ModeDescription = m_ModeDesc;

            /// link the SampleDescription
            m_SwapDesc.SampleDescription = m_SampleDesc;

            /// Number of buffers (including the front buffer)
            m_SwapDesc.BufferCount = 1;
            m_SwapDesc.Flags = SwapChainFlags.None;
            m_SwapDesc.IsWindowed = true;

            /// The output window (the windows being rendered to)
            m_SwapDesc.OutputHandle = this.m_RenderControl;

            /// Scrap the contents of the buffer every frame
            m_SwapDesc.SwapEffect = SwapEffect.Discard;

            /// Indicate that this SwapChain 
            /// is going to be a Render target
            m_SwapDesc.Usage = Usage.RenderTargetOutput;

            ///////////////////////////////////////////////////////////////////////// Create Chain

            // release the old chain 
            if (m_SwapChain != null)
                m_SwapChain.Dispose();

            /// Create the actual swap chain
            m_SwapChain = new SwapChain(Engine.g_factory, Engine.g_device, m_SwapDesc);

            ///////////////////////////////////////////////////////////////////////// Set the Render Target

            m_RenderStencil = Texture2D.FromSwapChain<Texture2D>(m_SwapChain, 0);

            /// Create and set the Render target - 
            /// the surface that we're actually going to draw on
            m_RenderTarget = new RenderTargetView(
                Engine.g_device, m_RenderStencil);

            ///////////////////////////////////////////////////////////////////////// Set the m_DepthStencil and m_DepthStencilView

            /// Create depth stencil texture
            /// and setup its settings
            Texture2DDescription descDepth = new Texture2DDescription();
            descDepth.Width = Width;
            descDepth.Height = Height;
            descDepth.MipLevels = 1;
            descDepth.ArraySize = 1;
            descDepth.Format = Format.D32_Float;
            descDepth.SampleDescription = new SampleDescription(1, 0);
            descDepth.Usage = ResourceUsage.Default;
            descDepth.BindFlags = BindFlags.DepthStencil;
            descDepth.CpuAccessFlags = CpuAccessFlags.None;
            descDepth.OptionFlags = ResourceOptionFlags.None;

            /// create the depth stencil view
            /// and setup its settings
            DepthStencilViewDescription descDSV = new DepthStencilViewDescription();
            descDSV.Format = descDepth.Format;
            descDSV.Dimension = DepthStencilViewDimension.Texture2D;
            //descDSV.MipSlice = 0;

            m_DepthStencil = new Texture2D(Engine.g_device, descDepth);
            m_DepthStencilView = new DepthStencilView(Engine.g_device, m_DepthStencil, descDSV);
        }

        /// <summary>
        /// Enable the rendering in this viewport
        /// </summary>
        public void EnableViewport(DeviceContext devCont)
        {
            // Set the render target to the specific viewport
            devCont.OutputMerger.SetTargets( m_DepthStencilView, m_RenderTarget );

            /// Clear or fill the Render target with a single color
            devCont.ClearRenderTargetView(
                m_RenderTarget, new Color4( 0.0f, 0.0f, 0.0f, 1.0f ) );
                //m_RenderTarget, new Color4( 0.0f, 0.125f, 0.3f, 1.0f ) );

            /// clear the depth buffer to 1.0 (max depth)
            devCont.ClearDepthStencilView(
                m_DepthStencilView, DepthStencilClearFlags.Depth,
                1.0f, 0 );

            /// Set the selected camera to the 
            /// camera of the selected viewport
            Engine.g_RenderCamera = m_Camera;

            /// Apply the viewport the rasterizer 
            /// (the thing that actually draws the triangle)
            devCont.Rasterizer.SetViewport( m_Viewport.X, m_Viewport.Y,
                m_Viewport.Width, m_Viewport.Height, m_Viewport.MinZ, m_Viewport.MaxZ);

            if ( m_RasterizerState.Description.FillMode != Settings.FillMode || m_RasterizerState.Description.CullMode != Settings.CullMode) {
                m_RasterizerStateDesc = new RasterizerStateDescription();
                m_RasterizerStateDesc.CullMode = Settings.CullMode;
                m_RasterizerStateDesc.FillMode = Settings.FillMode;
                m_RasterizerStateDesc.IsDepthClipEnabled = true;
                m_RasterizerStateDesc.IsScissorEnabled = false;
                m_RasterizerStateDesc.IsMultisampleEnabled = false;
                m_RasterizerStateDesc.IsAntialiasedLineEnabled = false;
                m_RasterizerStateDesc.SlopeScaledDepthBias = 0.0f;
                m_RasterizerStateDesc.DepthBiasClamp = 0.0f;
                m_RasterizerStateDesc.DepthBias = 0;
                

                m_RasterizerState = new RasterizerState( Engine.g_device, m_RasterizerStateDesc );
            }
            /// set the rasterizer state
            devCont.Rasterizer.State = m_RasterizerState;
        }

        /// <summary>
        /// Display the result to the screen
        /// </summary>
        public void Display()
        {
            /// Tell the system to display the new buffer
            m_SwapChain.Present(Settings.VSync, PresentFlags.None);
        }

        /// <summary>
        /// Resize the viewport
        /// </summary>
        /// <param name="Width"></param>
        /// <param name="Height"></param>
        public void Resize(int Width, int Height)
        {
            // dispose the buffers
            m_RenderTarget.Dispose();
            m_SwapChain.Dispose();

            // recreate the buffers
            SetView(Width, Height);

            /// change the viewport size
            m_Viewport.Width = Width;
            m_Viewport.Height = Height;

            // change the camera 
            m_Camera.SetProjParams(Width, Height, Settings.FOV, Width / (float)Height, 1.0f, Settings.NearPlane);
        }
    }
}

namespace GraphicsEngine.Core
{
    public struct CamViewport
    {
        public float X;
        public float Y;
        public float Width;
        public float Height;
        public float MinZ;
        public float MaxZ;
    }
}
