using System;
using System.Runtime.InteropServices;
using System.Windows.Forms;

/// library for directX
using SharpDX;
using SharpDX.Direct3D11;
using SharpDX.DXGI;
using SharpDX.DirectInput;
using SharpDX.RawInput;

/// resolve conflicts
using Device = SharpDX.Direct3D11.Device;

/// internal libraries
using GraphicsEngine.Core;
using System.Collections.Generic;
using GraphicsEngine.Managers;

using FxMaths;
using SharpDX.Direct3D;

namespace GraphicsEngine {
    public class Engine {

        #region OS integration
        /// A function that lets us peeks at the first message avaliable from the OS
        [DllImport("User32.dll", CharSet = CharSet.Auto)]
        static extern bool PeekMessage(out GraphicsEngine.Core.Message msg, IntPtr hWnd, uint messageFilterMin, uint messageFilterMax, uint flags);

        /// <summary>
        /// Checks is there are any messages 
        /// avaliable from the OS
        /// </summary>
        static bool IsIdle
        {
            get
            {
                GraphicsEngine.Core.Message msg;
                return !PeekMessage(out msg, IntPtr.Zero, 0, 0, 0);
            }
        }

        #endregion


        #region Variables

        public static Device g_device;

        #region private variables for render
        public static Factory g_factory;

        static ModeDescription g_modeDesc;
        static SampleDescription g_sampleDesc;
        static SwapChainDescription g_swapDesc;
        static SwapChain g_swapChain;
        #endregion

        /// <summary>
        /// Indicates if engine is running
        /// </summary>
        protected static bool g_Running = false;

        /// <summary>
        /// Camera used in engine to Render from
        /// </summary>
        public static Camera g_RenderCamera;

        /// <summary>
        /// Camera used in engine to Render from
        /// </summary>
        public static Camera g_MoveCamera;


        /// <summary>
        /// Device for deferred render
        /// </summary>
        public static DeviceContext DeferredDev1;
        public static DeviceContext DeferredDev2;


        #region Input Variables
        enum InputStateEnum { Acquire, Unacquire };

        static Keyboard g_keyboard;

        static Mouse g_mouse;

        static InputStateEnum MouseState;

        static InputStateEnum KeyboardState;

        /// <summary>
        /// The Control that we take the user input
        /// </summary>
        private Control InputControl;
        #endregion

        /// <summary>
        /// The handle of the form 
        /// </summary>
        private IntPtr FormHandle;

        /// <summary>
        /// The form that we using for rendering
        /// </summary>
        private Form form;

        #region Mesh Manager

        public static Object3DManager g_MeshManager;

        #endregion

        #endregion


        #region Properties
        /// <summary>
        /// Indicates if at least 
        /// one engine instance is running
        /// </summary>
        public static bool isEngineRunning
        {
            get { return g_Running; }
        }

        /// <summary>
        /// Indicate the number of drawing triangles
        /// </summary>
        public int NumTriangles
        {
            get
            {
                int sum=0;

                // add all the polygons of the scenes
                foreach ( GraphicsEngine.Core.PrimaryObjects3D.Mesh mesh in g_MeshManager.g_Object3DList ) {
                    sum += mesh.m_polygons.Count; 
                }

                return sum;
            }
        }
        #endregion


        #region Constructors

        /// <summary>
        /// Initializes a new engine, sets view , tex manager
        /// camera , mesh manager , texture manager , line manager
        /// volume manager
        /// </summary>
        /// <param name="width">The width of the Render target</param>
        /// <param name="height">The height of the Render target</param>
        public Engine(int Width, int Height, Form form)
        {
            // pass the settings of resolution
            Settings.Resolution = new System.Drawing.Size(Width, Height);

            /// set the handles for full screen or windows 
            this.form = form;
            this.FormHandle = form.Handle;

            /// Create the factory which manages general graphics resources
            g_factory = new Factory();
            g_factory.MakeWindowAssociation(this.FormHandle, WindowAssociationFlags.IgnoreAll);

            // find correct adapter
            int adapterCount = g_factory.GetAdapterCount();
            //MessageBox.Show(adapterCount.ToString());

            // we try to select the PerfHUD adapter 
            for (int i = 0; i < adapterCount; i++) {
                Adapter adapt = g_factory.GetAdapter(i);
                //MessageBox.Show(adapt.Description.Description);

                if (adapt.Description.Description == "NVIDIA PerfHUD") {
                    g_device = new Device(
                        adapt,
                        DeviceCreationFlags.Debug );
                }

                Console.WriteLine(i.ToString() + adapt.Description.Description);
            }

            if (g_device == null)
            {


                /// Create the DirectX Device
                g_device = new Device(g_factory.GetAdapter(0),
                                      (Settings.Debug) ? DeviceCreationFlags.Debug : DeviceCreationFlags.None,
                                      new FeatureLevel[] { FeatureLevel.Level_11_0 });

                /*
                                g_device = new Device(DriverType.Reference,
                                                        (Settings.Debug) ? DeviceCreationFlags.Debug : DeviceCreationFlags.None,
                                                        new FeatureLevel[] { FeatureLevel.Level_11_0 });

                 * */
                // check if we have one device to our system
                if (!(((g_device.FeatureLevel & FeatureLevel.Level_10_0) != 0) || ((g_device.FeatureLevel & FeatureLevel.Level_10_1) != 0) || ((g_device.FeatureLevel & FeatureLevel.Level_11_0) != 0)))
                {
                    // if we don't have we just simulate
                    #region Create the device base on swapChain
                    /// Create a description of the display mode
                    g_modeDesc = new ModeDescription();
                    /// Standard 32-bit RGBA
                    g_modeDesc.Format = Format.R8G8B8A8_UNorm;
                    /// Refresh rate of 60Hz (60 / 1 = 60)
                    g_modeDesc.RefreshRate = new Rational(60, 1);
                    /// Default
                    g_modeDesc.Scaling = DisplayModeScaling.Unspecified;
                    g_modeDesc.ScanlineOrdering =  DisplayModeScanlineOrder.Progressive;

                    /// ClientSize is the size of the
                    /// form without the title and borders
                    g_modeDesc.Width = Width;
                    g_modeDesc.Height = Height;

                    /// Create a description of the samping 
                    /// for multisampling or antialiasing
                    g_sampleDesc = new SampleDescription();
                    /// No multisampling
                    g_sampleDesc.Count = 1;
                    g_sampleDesc.Quality = 0;

                    /// Create a description of the swap 
                    /// chain or front and back buffers
                    g_swapDesc = new SwapChainDescription();
                    /// link the ModeDescription
                    g_swapDesc.ModeDescription = g_modeDesc;
                    /// link the SampleDescription
                    g_swapDesc.SampleDescription = g_sampleDesc;
                    /// Number of buffers (including the front buffer)
                    g_swapDesc.BufferCount = 1;
                    g_swapDesc.Flags = SwapChainFlags.AllowModeSwitch;
                    g_swapDesc.IsWindowed = true;
                    /// The output window (the windows being rendered to)
                    g_swapDesc.OutputHandle = this.FormHandle;
                    /// Scrap the contents of the buffer every frame
                    g_swapDesc.SwapEffect = SwapEffect.Discard;
                    /// Indicate that this SwapChain 
                    /// is going to be a Render target
                    g_swapDesc.Usage = Usage.RenderTargetOutput;


                    //g_swapChain = new SwapChain(g_factory, g_device, g_swapDesc);

                    try
                    {
                        /// Create the actual swap chain
                        /// Here we set and the device type
                        Device.CreateWithSwapChain(DriverType.Warp, (Settings.Debug) ? DeviceCreationFlags.Debug : DeviceCreationFlags.None, new FeatureLevel[] { Settings.FeatureLevel }, g_swapDesc, out g_device, out g_swapChain);
                    }
                    catch (Exception ex)
                    {
                        /// Create the actual swap chain
                        /// Here we set and the device type
                        Device.CreateWithSwapChain(DriverType.Reference, (Settings.Debug) ? DeviceCreationFlags.Debug : DeviceCreationFlags.None, new FeatureLevel[] { Settings.FeatureLevel }, g_swapDesc, out g_device, out g_swapChain);
                    }

                    /// Create the factory which manages general graphics resources
                    g_factory = g_swapChain.GetParent<Factory>();

                    g_factory.MakeWindowAssociation(this.FormHandle, WindowAssociationFlags.IgnoreAll);
                    #endregion
                }
                else
                {
#if false

                    #region Create the device base on swapChain
                    /// Create a description of the display mode
                    g_modeDesc = new ModeDescription();
                    /// Standard 32-bit RGBA
                    g_modeDesc.Format = Format.R8G8B8A8_UNorm;
                    /// Refresh rate of 60Hz (60 / 1 = 60)
                    g_modeDesc.RefreshRate = new Rational(60, 1);
                    /// Default
                    g_modeDesc.Scaling = DisplayModeScaling.Unspecified;
                    g_modeDesc.ScanlineOrdering = DisplayModeScanlineOrdering.Progressive;

                    /// ClientSize is the size of the
                    /// form without the title and borders
                    g_modeDesc.Width = Width;
                    g_modeDesc.Height = Height;

                    /// Create a description of the samping 
                    /// for multisampling or antialiasing
                    g_sampleDesc = new SampleDescription();
                    /// No multisampling
                    g_sampleDesc.Count = 1;
                    g_sampleDesc.Quality = 0;

                    /// Create a description of the swap 
                    /// chain or front and back buffers
                    g_swapDesc = new SwapChainDescription();
                    /// link the ModeDescription
                    g_swapDesc.ModeDescription = g_modeDesc;
                    /// link the SampleDescription
                    g_swapDesc.SampleDescription = g_sampleDesc;
                    /// Number of buffers (including the front buffer)
                    g_swapDesc.BufferCount = 1;
                    g_swapDesc.Flags = SwapChainFlags.None;
                    g_swapDesc.IsWindowed = true;
                    /// The output window (the windows being rendered to)
                    g_swapDesc.OutputHandle = FormHandle;
                    /// Scrap the contents of the buffer every frame
                    g_swapDesc.SwapEffect = SwapEffect.Discard;
                    /// Indicate that this SwapChain 
                    /// is going to be a Render target
                    g_swapDesc.Usage = Usage.RenderTargetOutput;

                    g_swapChain = new SwapChain(g_factory, g_device, g_swapDesc);

                    #endregion 
#endif
                }

                // set the feature level
                Settings.FeatureLevel = g_device.FeatureLevel;
            }
            

            /// init mesh manager 
            g_MeshManager = new Object3DManager();

            /// init deferred device
            //DeferredDev1 = new DeviceContext( Engine.g_device );
            //DeferredDev2 = new DeviceContext( Engine.g_device );

            /// Set flag to indicate that engine is running
            g_Running = true;

            Logger.WriteLine( "The Graphing Engine have be start " );

            // set the event handler
            RenderLoopHandler = new EventHandler( RenderLoop );
        }

        #endregion


        #region Functions
        /// <summary>
        /// called when destroying the directx
        /// device. It disposes all textures,
        /// structures, meshes and resources used 
        /// by the engine
        /// </summary>
        public void Dispose()
        {
            // dispose the directX part 
            g_device.Dispose();
            g_swapChain.Dispose();
            g_factory.Dispose();

            g_Running = false;
        }

        #endregion


        #region Input Functions

        private void Release_KeyBoard()
        {
            if (KeyboardState != InputStateEnum.Unacquire)
            {
                g_keyboard.Unacquire();
                KeyboardState = InputStateEnum.Unacquire;
            }
        }

        private void Release_Mouse()
        {
            if (MouseState != InputStateEnum.Unacquire)
            {
                g_mouse.Unacquire();
                g_mouse.SetCooperativeLevel(InputControl,
                                             CooperativeLevel.NonExclusive |
                                             CooperativeLevel.Foreground
                                           );
                MouseState = InputStateEnum.Unacquire;
            }
        }


        /// <summary>
        /// Initialize input methods and
        /// setup it's settings
        /// </summary>
        /// <param name="_control"></param>
        public void SetupInput(Control _control)
        {

            InputControl = _control;
            /// make sure that DirectInput has been initialized
            DirectInput dinput = new DirectInput();
            /// create the device
            try {
                g_keyboard = new Keyboard(dinput);
                g_keyboard.Properties.BufferSize = 256;

                g_keyboard.SetCooperativeLevel( _control, 
                                                CooperativeLevel.Foreground |
                                                CooperativeLevel.NonExclusive |
                                                CooperativeLevel.NoWinKey);

                g_keyboard.Acquire();
            } catch (MarshalDirectiveException e) {
                MessageBox.Show(e.Message);
                return;
            }

            /// acquire the device
            Release_KeyBoard();

            /// setup Mouse
            dinput = new DirectInput();
            try {
                g_mouse = new Mouse(dinput);
                g_mouse.SetCooperativeLevel(_control,
                    CooperativeLevel.Exclusive | CooperativeLevel.Foreground);
            } catch (MarshalDirectiveException e) {
                MessageBox.Show(e.Message);
                return;
            }

            /// since we want to use buffered data, 
            /// we need to tell DirectInput
            /// to set up a buffer for the data
            g_mouse.Properties.BufferSize = 10;

            /// acquire the device and release to OS
            Release_Mouse();

        }

        /// <summary>
        /// Release input devices,
        /// keyboard and mouse from the Render engine
        /// so we can move in Render form,
        /// used in unfocus
        /// </summary>
        public void releaseInput()
        {
            Release_KeyBoard();
            Release_Mouse();
        }

        /// <summary>
        /// Called to make the Render engine
        /// acquire the input devices (mouse and keyboard)
        /// </summary>
        public void refocusInput()
        {
            if (KeyboardState != InputStateEnum.Acquire)
            {
                g_keyboard.Acquire();
                KeyboardState = InputStateEnum.Acquire;
            }

            if (MouseState != InputStateEnum.Acquire)
            {
                g_mouse.Unacquire();
                g_mouse.SetCooperativeLevel(InputControl,
                                             CooperativeLevel.Exclusive |
                                             CooperativeLevel.Foreground
                                           );
                g_mouse.Properties.AxisMode = DeviceAxisMode.Relative;
                g_mouse.Acquire();

                MouseState = InputStateEnum.Acquire;
            }
        }

        /// <summary>
        /// Read from keyboard and mouse
        /// and pass actions and keys to handlers
        /// </summary>
        private void readInput(Camera cam)
        {

            // if keyboard is acquired
            if (KeyboardState == InputStateEnum.Acquire)
            {
                KeyboardState key_state;

                //poll
                try
                {
                    g_keyboard.Poll();
                }
                catch (Exception e)
                {
                    if (Result.GetResultFromException(e).Failure == true)
                        goto mouse_proccess;
                }

                // get keyboard state
                key_state = g_keyboard.GetCurrentState();

                /// combine the keys pressed to create 
                /// a move vector and pass
                foreach (Key key in key_state.PressedKeys)
                {
                    cam.handleKeys(key);
                    if (key == Key.Escape)
                        releaseInput();
                }
            }

        mouse_proccess:

            // if the engine owns the mouse device
            if (MouseState == InputStateEnum.Acquire)
            {
                MouseState mouse_state;

                try
                {
                    g_mouse.Poll();
                }
                catch (Exception e)
                {
                    if (Result.GetResultFromException(e).Failure == true)
                        return;
                }

                // get state
                mouse_state = g_mouse.GetCurrentState();

                // get list of mouse data
                var bufferedData = g_mouse.GetBufferedData();

                //fail check
                if (bufferedData == null)
                    goto mouse_exit;

                //fail check
                if (bufferedData.Length == 0)
                    goto mouse_exit;

                //counters
                float cnt_x = 0;
                float cnt_y = 0;
                float cnt_z = 0;

                // combine the mouse buffered data to create a single mouse state
                foreach (var packet in bufferedData)
                {
                    //handle event
                    switch (packet.Offset)
                    {
                        case MouseOffset.X:
                            mouse_state.X += (int)(packet.Value);
                            cnt_x++;
                            break;
                        case MouseOffset.Y:
                            mouse_state.Y += (int)(packet.Value);
                            cnt_y++;
                            break;
                        case MouseOffset.Z:
                            mouse_state.Z += (int)(packet.Value);
                            cnt_z++;
                            break;
                    }
                }

                //normalize..
                if (cnt_x > 0) mouse_state.X = (int)(mouse_state.X / cnt_x); else mouse_state.X = 0;
                if (cnt_y > 0) mouse_state.Y = (int)(mouse_state.Y / cnt_y); else mouse_state.Y = 0;
                if (cnt_z > 0) mouse_state.Z = (int)(mouse_state.Z / cnt_z); else mouse_state.Z = 0;

                /// pass the mouse state to camera handler 
                cam.handleMouse(mouse_state);
            }
            else
            {
                cam.handleMouse(new MouseState());
            }

        mouse_exit:
            cam.handleMouse(new MouseState());

        }
        #endregion


        #region Timing Variables
        static float TimeElapsed = 0.0f;
        static long TimeStart = 0;
        static long FPS_Count = 0;
        static long OldFPS = 0;
        static long TimeFPSStart = 0;
        static long TimeCurrent;

        /// <summary>
        /// The FPS of the previous second 
        /// </summary>
        public long FPS
        {
            get { return OldFPS; }
        }
        #endregion


        #region Events
        
        private event EventHandler RenderLoopHandler;

        /// <summary>
        /// Start the render
        /// This must run after the setup of all 
        /// necessary components.
        /// </summary>
        public void StartRender()
        {
            /// When the application is idle 
            /// (has handled all system messages) the event is raised
            Application.Idle += RenderLoopHandler;
        }

        /// <summary>
        /// Stop the render
        /// </summary>
        public void StopRender()
        {
            // remove the handler
            Application.Idle -=RenderLoopHandler;
        }

        /// <summary>
        /// The Render loop , runs while there are no 
        /// message pending from the OS. Once they are server
        /// the loop is called again
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        void RenderLoop( object sender, EventArgs e )
        {
            /// While there are no system messages, keep rendering
            while ( IsIdle ) {
                /// find the time elapsed 
                /// get current time
                TimeCurrent = DateTime.Now.TimeOfDay.Ticks;
                if ( TimeStart == 0.0f )
                    TimeStart = TimeCurrent;
                /// find how long the app is running
                TimeElapsed = ( TimeCurrent - TimeStart ) / 1000.0f;
                /// display Frame Rate messages on screen
                /// once every a time period
                if ( TimeCurrent - TimeFPSStart >= System.TimeSpan.TicksPerSecond ) {
                    /// TODO: show FPS message

                    /// reset Timer base
                    TimeFPSStart = TimeCurrent;

                    OldFPS = FPS_Count;
                    FPS_Count = 0;
                } else {
                    /// increase FPS counter
                    FPS_Count++;
                }

                TimeStart = TimeCurrent;

                /// read the input from the keyboard
                readInput( g_MoveCamera );

                /// move the camera
                g_MoveCamera.FrameMove( TimeElapsed );


                //DeviceContext selectedContext  = Engine.DeferredDev;

                DeviceContext selectedContext  = Engine.g_device.ImmediateContext;
                //DeviceContext devCont1 = Engine.DeferredDev1;
                //DeviceContext devCont2 = Engine.DeferredDev2;
                DeviceContext []devCont = { selectedContext };
                foreach ( Core.Viewport viewport in ViewportManager.ListOfViewport ) {

                    //Performance.BeginEvent( new Color4( System.Drawing.Color.Red ), "EnableViewport" );
                    // enable the specific viewport
                    viewport.EnableViewport( selectedContext );
                    //viewport.EnableViewport( devCont1 );
                    //Performance.EndEvent();


                    //Performance.BeginEvent( new Color4( System.Drawing.Color.Yellow ), "PreProcessing" );
                    /// call all cb
                    RenderCBManager.RunCB( RenderCBTiming.PreProcessing );
                    //Performance.EndEvent();


                    //Performance.BeginEvent( new Color4( System.Drawing.Color.Blue ), "Render" );
                    
                    /// render mesh list
                    g_MeshManager.Render( devCont );
                    //Performance.EndEvent();


                    //Performance.BeginEvent( new Color4( System.Drawing.Color.Yellow ), "PostProcessing" );
                    /// call all cb
                    RenderCBManager.RunCB( RenderCBTiming.PostProcessing );
                    //Performance.EndEvent();


                    //Performance.BeginEvent( new Color4( System.Drawing.Color.Red ), "ExecuteCommandList" );

                    //CommandList df_cl = devCont1.FinishCommandList( false );
                    //Engine.g_device.ImmediateContext.ExecuteCommandList( df_cl, true );

                    //df_cl = devCont2.FinishCommandList( false );
                    //Engine.g_device.ImmediateContext.ExecuteCommandList( df_cl, true );

                    //Performance.EndEvent();



                    //Performance.BeginEvent( new Color4( System.Drawing.Color.Yellow ), "Display" );
                    // display the result
                    viewport.Display();
                    //Performance.EndEvent();
                }
            }
        }
        #endregion

        
    }
}
