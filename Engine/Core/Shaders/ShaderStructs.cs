using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;


using System.Runtime.InteropServices;
using SharpDX;

namespace GraphicsEngine.Core.Shaders
{

    [StructLayout( LayoutKind.Explicit, Size = 524 )] // 396
    public struct cbViewMatrix
    {
        /// <summary>
        /// Transform object vertices to world-space:
        /// </summary>
        [FieldOffset( 0 )]
        public Matrix g_mWorld;

        /// <summary>
        /// Transform object vertices to view space and project them in perspective:
        /// matrix with World*View*Projection
        /// </summary>
        [FieldOffset( 64 )]
        public Matrix g_mWorldViewProjection;

        /// <summary>
        /// matrix with World* View 
        /// </summary>
        [FieldOffset( 128 )]
        public Matrix g_mWorldView;

        /// <summary>
        /// matrix with View 
        /// </summary>
        [FieldOffset( 192 )]
        public Matrix g_mView;

        /// <summary>
        /// matrix with inverse world trans
        /// </summary>
        [FieldOffset( 256 )]
        public Matrix g_mWorldInverseTrans;

        /// <summary>
        /// inverse view matrix
        /// </summary>
        [FieldOffset( 320 )]
        public Matrix g_mViewInverse;

        /// <summary>
        /// matrix with Projection
        /// </summary>
        [FieldOffset( 384 )]
        public Matrix g_mProjection;

        /// <summary>
        /// matrix with View * Projection
        /// </summary>
        [FieldOffset(448)]
        public Matrix g_mViewProjection;

        /// <summary>
        /// the position of the camera 
        /// </summary>
        [FieldOffset( 512 )]
        public Vector3 g_vCameraPosition;

    }




    [StructLayout( LayoutKind.Explicit, Size = 64 )] //56
    public struct cbMaterial
    {
        
        /// <summary>
        /// Material's ambient color
        /// </summary>
        [FieldOffset( 0 )]
        public Vector3 g_vMaterialAmbient;

        /// <summary>
        /// Material's diffuse color
        /// </summary>
        [FieldOffset( 12 )]
        public Vector3 g_vMaterialDiffuse;

        /// <summary>
        /// Material's specular color
        /// </summary>
        [FieldOffset( 24 )]
        public Vector3 g_vMaterialSpecular;

        /// <summary>
        /// Transparency of the material
        /// </summary>
        [FieldOffset( 36 )]
        public float  g_fMaterialAlpha;

        /// <summary>
        /// How shiny the material is 
        /// </summary>
        [FieldOffset( 40 )]
        public float g_nMaterialShininess;

        [FieldOffset( 44 )]
        public float Ks;

        [FieldOffset( 48 )]
        public float Eccentricity;

        [FieldOffset( 52 )]
        public float Kr;

    }
}
