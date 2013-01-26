using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpDX;
using SharpDX.DirectInput;
using System.Drawing;
using SharpDX.Direct3D11;

namespace GraphicsEngine.Core {

    public class Camera {
        #region Private Variables
        /// <summary>
        /// Mapped camera actions
        /// </summary>
        private enum CameraKeys {
            STRAFE_LEFT = 0,
            STRAFE_RIGHT,
            MOVE_FORWARD,
            MOVE_BACKWARD,
            MOVE_UP,
            MOVE_DOWN,
            RESET,
            MAX_KEYS,
            UNKNOWN = 0xFF
        }
        /// <summary>
        /// Direction vector of keyboard input
        /// </summary>
        private Vector3 m_KeyboardDirection = new Vector3();
        /// <summary>
        /// Rotation Velocity of camera
        /// </summary>
        private Vector2 m_RotVelocity;
        /// <summary>
        /// Velocity of camera
        /// </summary>
        private Vector3 m_Velocity;
        /// <summary>
        /// View matrix
        /// </summary>
        private Matrix m_View;
        /// <summary>
        /// Projection matrix
        /// </summary>
        private Matrix m_Proj;
        /// <summary>
        /// Default camera eye position
        /// </summary>
        private Vector3 m_DefaultEye;
        /// <summary>
        /// Default LookAt position
        /// </summary>
        private Vector3 m_DefaultLookAt;
        /// <summary>
        /// Camera eye position
        /// </summary>
        private Vector3 m_Eye;
        /// <summary>
        /// LookAt position
        /// </summary>
        private Vector3 m_LookAt;
        /// <summary>
        /// Yaw angle of camera
        /// </summary>
        private float m_CameraYawAngle;
        /// <summary>
        /// Pitch angle of camera
        /// </summary>
        private float m_CameraPitchAngle;
        /// <summary>
        /// Windows Width
        /// </summary>
        private int m_Windows_Width;
        /// <summary>
        /// Windows Height
        /// </summary>
        private int m_Windows_Height;
        /// <summary>
        /// Field of view
        /// </summary>
        private float m_FOV;
        /// <summary>
        /// Aspect ratio
        /// </summary>
        private float m_Aspect;
        /// <summary>
        /// Near plane
        /// </summary>
        private float m_NearPlane;
        /// <summary>
        /// Far plane
        /// </summary>
        private float m_FarPlane;
        /// <summary>
        /// Scaler for rotation
        /// </summary>
        private float m_RotationScaler;
        /// <summary>
        /// Scaler for movement
        /// </summary>
        private float m_MoveScaler;

        // Mouse
        /// <summary>
        /// Mouse relative delta smoothed over a few frames
        /// </summary>
        private Vector2 m_MouseDelta;
        /// <summary>
        /// Number of frames to smooth mouse data over
        /// </summary>
        private float m_FramesToSmoothMouseData;
        #endregion

        #region Properties
        public Matrix ViewMatrix { get { return m_View; } }

        public Matrix ProjMatrix { get { return m_Proj; } }

        public Vector3 Eye { get { return m_Eye; } set { SetViewParams(value, m_LookAt); } }

        public Vector3 LookAt { get { return m_LookAt; } set { SetViewParams(m_Eye, value); } }

        public float NearClip { get { return m_NearPlane; } }

        public float FarClip { get { return m_FarPlane; } }

        public Vector3 WorldUp
        {
            get
            {
                return Vector3.TransformCoordinate(new Vector3(0, 1, 0), Matrix.RotationYawPitchRoll(
                    m_CameraYawAngle, m_CameraPitchAngle, 0));
            }
        }

        public int WindowsWidth { get { return m_Windows_Width; } }
        public int WindowsHeight { get { return m_Windows_Height; } }
        public Vector3 Direction { get { return m_LookAt - m_Eye; } }
        #endregion

        #region Constructors
        /// <summary>
        /// 
        /// </summary>
        /// <param name="width">Width of the Render target</param>
        /// <param name="height">Height of the Render target</param>
        public Camera(int width, int height)
        {
            // Set attributes for the view matrix
            Vector3 Eye = new Vector3(0.0f, 1.0f, -10.0f);
            Vector3 Lookat = new Vector3(0.0f, 1.0f, 0.0f);

            // Setup the view matrix
            SetViewParams(Eye, Lookat);

            // Setup the projection matrix
            SetProjParams(width, height, Settings.FOV, width / (float)height, 1.0f,  Settings.NearPlane);

            m_CameraYawAngle = 0.0f;
            m_CameraPitchAngle = 0.0f;

            m_Velocity = new Vector3(0, 0, 0);
            m_RotVelocity = new Vector2(0, 0);

            m_RotationScaler = 0.002f;
            m_MoveScaler = 0.3f;

            m_MouseDelta = new Vector2(0, 0);
            m_FramesToSmoothMouseData = 8.0f;

            m_Windows_Width = width;
            m_Windows_Height = height;
        }
        #endregion

        #region Public Functions
        /// <summary>
        /// Client can call this to change the position and direction of camera
        /// </summary>
        /// <param name="eye">Position of the camera</param>
        /// <param name="LookAt">Where the camera looks at</param>
        public void SetViewParams(Vector3 eye, Vector3 LookAt)
        {
            if (eye == null || LookAt == null)
                return;

            m_DefaultEye = m_Eye = eye;
            m_DefaultLookAt = m_LookAt = LookAt;

            // calc the view matrix
            Vector3 up = new Vector3(0.0f, 1.0f, 0.0f);
            m_View = Matrix.LookAtLH(m_Eye, m_LookAt, up);

            Matrix InvView;
            InvView = Matrix.Invert(m_View);

            /// The axis basis vectors and camera position are stored inside the 
            /// position matrix in the 4 rows of the camera's world matrix.
            /// To figure out the yaw/pitch of the camera, we just need the Z basis vector
            Vector3 ZBase = new Vector3(InvView.M31, InvView.M32, InvView.M33);

            m_CameraYawAngle = (float)Math.Atan2(ZBase.X, ZBase.Z);
            float Len = (float)Math.Sqrt(ZBase.Z * ZBase.Z + ZBase.Z * ZBase.X);
            m_CameraPitchAngle = -(float)Math.Atan2(ZBase.Y, Len);
        }

        /// <summary>
        /// Set attributes for the projection matrix
        /// </summary>
        /// <param name="width">Width of the Render target</param>
        /// <param name="height">Height of the Render target</param>
        /// <param name="FOV">Field of view in degrees</param>
        /// <param name="Aspect"></param>
        /// <param name="NearPlane"></param>
        /// <param name="FarPlane"></param>
        public void SetProjParams(
            int width, int height, float FOV,
            float Aspect, float NearPlane,
            float FarPlane)
        {
            m_FOV = FOV;
            m_Aspect = Aspect;
            m_NearPlane = NearPlane;
            m_FarPlane = FarPlane;
            m_Windows_Width = width;
            m_Windows_Height = height;
            m_Proj = Matrix.PerspectiveFovLH(
                FOV, Aspect, NearPlane, FarPlane);
        }

        /// <summary>
        /// Reset the camera's position back to the default
        /// </summary>
        public void Reset()
        {
            SetViewParams(m_DefaultEye, m_DefaultLookAt);
        }

        /// <summary>
        /// Move the camera based on 
        /// keyboard and mouse input,
        /// elapsed time and elapsed time since 
        /// last move. The last part makes sure
        /// movement is always smooth regardless of 
        /// frame rate
        /// </summary>
        /// <param name="elapsedTime">Elapsed time since the last move</param>
        public void FrameMove(float elapsedTime)
        {
            /// Get amount of velocity based on 
            /// the keyboard input and drag (if any)
            UpdateVelocity(elapsedTime);

            /// Simple euler method to calculate position delta
            Vector3 PosDelta = m_Velocity * elapsedTime;

            /// rotating the camera
            /// Update the pitch & yaw angle based on mouse movement
            float YawDelta = m_RotVelocity.X;
            float PitchDelta = m_RotVelocity.Y;

            m_CameraPitchAngle += -PitchDelta;
            m_CameraYawAngle += YawDelta;

            /// Limit pitch to straight up or straight down
            m_CameraPitchAngle =
                Math.Max(-(float)Math.PI / 2.0f, m_CameraPitchAngle);
            m_CameraPitchAngle =
                Math.Min(+(float)Math.PI / 2.0f, m_CameraPitchAngle);

            /// Make a rotation matrix based on the camera's yaw & pitch
            Matrix CameraRot = Matrix.RotationYawPitchRoll(
                m_CameraYawAngle, m_CameraPitchAngle, 0);

            /// Transform vectors based on camera's rotation matrix
            Vector3 WorldUp, WorldAhead;
            Vector3 LocalUp = new Vector3(0, 1, 0);
            Vector3 LocalAhead = new Vector3(0, 0, 1);
            WorldUp = Vector3.TransformCoordinate(LocalUp, CameraRot);
            WorldAhead = Vector3.TransformCoordinate(LocalAhead, CameraRot);

            /// Transform the position delta by the camera's rotation 
            Vector3 PosDeltaWorld;
            PosDeltaWorld = Vector3.TransformCoordinate(PosDelta, CameraRot);

            /// Move the eye position 
            m_Eye += PosDeltaWorld;

            /// Update the lookAt position based on the eye position 
            m_LookAt = m_Eye + WorldAhead;

            /// Update the view matrix
            m_View = Matrix.LookAtLH(m_Eye, m_LookAt, WorldUp);
        }
        #endregion

        #region Private Function
        /// <summary>
        /// Maps the keyboard keys to camera actions
        /// </summary>
        /// <param name="key">Key from keyboard</param>
        /// <returns>Action to perform</returns>
        private static CameraKeys MapKey(System.Windows.Forms.Keys key)
        {
            switch (key) {
                case System.Windows.Forms.Keys.S:
                    return CameraKeys.MOVE_BACKWARD;
                case System.Windows.Forms.Keys.W:
                    return CameraKeys.MOVE_FORWARD;
                case System.Windows.Forms.Keys.A:
                    return CameraKeys.STRAFE_LEFT;
                case System.Windows.Forms.Keys.D:
                    return CameraKeys.STRAFE_RIGHT;
                case System.Windows.Forms.Keys.Q:
                    return CameraKeys.MOVE_UP;
                case System.Windows.Forms.Keys.E:
                    return CameraKeys.MOVE_DOWN;

                default:
                    return CameraKeys.UNKNOWN;
            }
        }

        /// <summary>
        /// Calculate the velocity direction and magnitude
        /// of the camera based on the vector created from
        /// keys pressed and the elapsed time since
        /// last update.
        /// </summary>
        /// <param name="ElapsedTime">Elapsed time since last camera move</param>
        private void UpdateVelocity(float ElapsedTime)
        {
            /// the Accelaration is the direction that we get from the keyboard
            Vector3 Accel = m_KeyboardDirection;
            /// reset the direction
            m_KeyboardDirection = Vector3.Zero;
            /// Normalize vector so if moving 2 dirs (left & forward), 
            /// the camera doesn't move faster than if moving in 1 dir
            Accel = Vector3.Normalize(Accel);
            /// Scale the acceleration vector
            Accel *= m_MoveScaler;
            /// the velocity to used in camera moce
            m_Velocity = Accel;
        }
        #endregion

        #region Handle Input

        /// <summary>
        /// Gets the input key and alters the move direction
        /// </summary>
        /// <param name="key">Input key from keyboard</param>
        public void handleKeys(System.Windows.Forms.Keys key)
        {
            /// Get mapped action
            CameraKeys mapKey = MapKey(key);
            float Steps=1f;
            /// add action to move vector
            switch (mapKey) {
                case CameraKeys.MOVE_FORWARD:
                    m_KeyboardDirection.Z += Steps;
                    break;
                case CameraKeys.MOVE_BACKWARD:
                    m_KeyboardDirection.Z -= Steps;
                    break;
                case CameraKeys.MOVE_UP:
                    m_KeyboardDirection.Y += Steps;
                    break;
                case CameraKeys.MOVE_DOWN:
                    m_KeyboardDirection.Y -= Steps;
                    break;
                case CameraKeys.STRAFE_RIGHT:
                    m_KeyboardDirection.X += Steps;
                    break;
                case CameraKeys.STRAFE_LEFT:
                    m_KeyboardDirection.X -= Steps;
                    break;
            }
        }

        /// <summary>
        /// Handles the mouse events
        /// </summary>
        /// <param name="mouse_state">The state of the mouse</param>
        public void handleMouse(Vector2 CurMouseDelta)
        {
            /// Smooth the relative mouse data over a few frames so it isn't 
            /// jerky when moving slowly at low frame rates.
            float fPercentOfNew = 1.0f / m_FramesToSmoothMouseData;
            float fPercentOfOld = 1.0f - fPercentOfNew;
            m_MouseDelta.X = m_MouseDelta.X * fPercentOfOld + CurMouseDelta.X * fPercentOfNew;
            m_MouseDelta.Y = m_MouseDelta.Y * fPercentOfOld + CurMouseDelta.Y * fPercentOfNew;
            m_RotVelocity = m_MouseDelta * m_RotationScaler;
        }
        #endregion

    }
}
