using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;

using SlimDX.Direct3D11;

namespace GraphicsEngine {

    public static class Settings {

        /// <summary>
        /// Sync the screen frame rate with card frames
        /// </summary>
        public static int VSync = 1;

        /// <summary>
        /// Resolution 
        /// </summary>
        public static Size Resolution;

        /// <summary>
        /// The application is in full screen
        /// </summary>
        public static Boolean isFullScreen = false;

        /// <summary>
        /// The frame rate when we use vSync
        /// </summary>
        public static int FrameRate = 60;

        /// <summary>
        /// The Fill mode of the viewports
        /// </summary>
        public static FillMode FillMode = SlimDX.Direct3D11.FillMode.Solid;

        /// <summary>
        /// The fill mode of the viewports
        /// </summary>
        public static CullMode CullMode = SlimDX.Direct3D11.CullMode.None;

        /// <summary>
        /// The driver will be set as a debug 
        /// </summary>
#if DEBUG
        public static Boolean Debug = true;
#else
        public static Boolean Debug = true;
#endif

        public static float NearPlane = 50000.0f;

        public static float FOV = (float)Math.PI / 4;


        /// <summary>
        /// The running Feature level.
        /// </summary>
        public static FeatureLevel FeatureLevel = SlimDX.Direct3D11.FeatureLevel.Level_11_0;

    }

    public class SettingsDummy
    {

        #region Properties

        /// <summary>
        /// Sync the screen frame rate with card frames
        /// </summary>
        public Boolean Use_VSync
        {
            get { return (Settings.VSync>0); }
            set { Settings.VSync = (value)?1:0; }
        }

        /// <summary>
        /// The frame rate when we use vSync
        /// </summary>
        public int Frame_Rate
        {
            get { return Settings.FrameRate; }
            set { Settings.FrameRate = value; }
        }

        /// <summary>
        /// The Fill mode of the viewports
        /// </summary>
        public FillMode Fill_Mode
        {
            get { return Settings.FillMode; }
            set { Settings.FillMode = value; }
        }

        /// <summary>
        /// The fill mode of the viewports
        /// </summary>
        public CullMode Cull_Mode
        {
            get { return Settings.CullMode; }
            set { Settings.CullMode = value; }
        }

        /// <summary>
        /// The driver will be set as a debug 
        /// </summary>
        public Boolean Debug
        {
            get { return Settings.Debug; }
            set { Settings.Debug = value; }
        }

        /// <summary>
        /// The running Feature level.
        /// </summary>
        public FeatureLevel FeatureLevel
        {
            get { return Settings.FeatureLevel; }
            set { Settings.FeatureLevel = value; }
        }

        /// <summary>
        /// The FOV division of PI
        /// </summary>
        public float FOV_piDiv
        {
            get { return (float)Math.PI/Settings.FOV; }
            set { Settings.FOV = (float)Math.PI / value; }
        }

        /// <summary>
        /// The Near plane cut
        /// </summary>
        public float NearPlane
        {
            get { return Settings.NearPlane; }
            set { Settings.NearPlane = value; }
        }

        #endregion
    }
}
