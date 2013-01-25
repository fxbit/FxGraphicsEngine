using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

// SlimDx lib
using SlimDX;

// internals libraries
using GraphicsEngine.Core;
using System.Security;

namespace GraphicsEngine.Import.FBX_Import {
    [StructLayout(LayoutKind.Sequential)]
    public struct Vector3_t {
        public float x, y, z;

        public override string ToString()
        {
            return "x=" + x.ToString() + "  y=" + y.ToString() + "  z=" + z.ToString();
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Vector2_t {
        public float u, v;

        public override string ToString()
        {
            return "u=" + v.ToString() + "  v=" + u.ToString();
        }
    }

    [SuppressUnmanagedCodeSecurityAttribute]
    internal static class FBX {
        const string FBX_Path = "Import/FBX/FBX_dll.dll";

        [DllImport(FBX_Path, CallingConvention = CallingConvention.Cdecl)]
        public static extern int fnFBX_dll();

        [DllImport(FBX_Path, CallingConvention = CallingConvention.Cdecl)]
        public static extern int Initialize(StringBuilder FileName);

        [DllImport(FBX_Path, CallingConvention = CallingConvention.Cdecl)]
        public static extern int GetNumberMesh();

        [DllImport(FBX_Path, CallingConvention = CallingConvention.Cdecl)]
        public static extern int SelectMesh(int MeshNum);

        [DllImport(FBX_Path, CallingConvention = CallingConvention.Cdecl)]
        public static extern void GetMeshName(StringBuilder Name);

        [DllImport(FBX_Path, CallingConvention = CallingConvention.Cdecl)]
        public static extern int GetPolygons_Count();

        [DllImport(FBX_Path, CallingConvention = CallingConvention.Cdecl)]
        public static extern Polygon GetPolygon(int i);

        [DllImport(FBX_Path, CallingConvention = CallingConvention.Cdecl)]
        public static extern Vector3_t GetPosition();

        [DllImport(FBX_Path, CallingConvention = CallingConvention.Cdecl)]
        public static extern Vector3_t GetRotation();

        [DllImport(FBX_Path, CallingConvention = CallingConvention.Cdecl)]
        public static extern Vector3_t GetScale();


        [DllImport(FBX_Path, CallingConvention = CallingConvention.Cdecl)]
        public static extern int GetMaterialCount();

        [DllImport(FBX_Path, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern int GetMaterialName(int materialID, StringBuilder Name);

        [DllImport(FBX_Path, CallingConvention = CallingConvention.Cdecl)]
        public static extern int GetTextureCount(int materialID);

        [DllImport(FBX_Path, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern void GetTexture(int materialID, int TextureID, StringBuilder PropertyName, StringBuilder RelativePath);

    }
}
