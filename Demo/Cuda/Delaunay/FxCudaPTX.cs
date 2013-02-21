using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using ManagedCuda;
using System.Reflection;
using System.IO;
using ManagedCuda.BasicTypes;

namespace Delaunay
{
    public class FxCudaPTX
    {
        Stream ptxFile = null;
        FxCuda cuda = null;

        public FxCudaPTX(FxCuda cuda, String filename, String path)
        {
            string resName;
            if (IntPtr.Size == 8)
                resName = filename+"x64.ptx";
            else
                resName = filename + ".ptx";

            ptxFile = File.OpenRead(path + "/" + resName);
            this.cuda = cuda;
        }

        public CudaKernel LoadKernel(String kernelName)
        {
            return cuda.ctx.LoadKernelPTX(ptxFile, kernelName);
        }

        public void Dispose()
        {
            ptxFile.Close();
            ptxFile.Dispose();
        }
    }
}
