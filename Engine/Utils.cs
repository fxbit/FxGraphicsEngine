using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpDX;

namespace GraphicsEngine
{
    public static class Utils
    {
        public static Matrix MatrixToGPU( Matrix mat )
        {
            return Matrix.Transpose( mat );
        }
    }
}
