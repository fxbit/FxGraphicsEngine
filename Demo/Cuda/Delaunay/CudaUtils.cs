using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using ManagedCuda;
using ManagedCuda.BasicTypes;
using System.Runtime.InteropServices;

namespace Delaunay
{
    public class CudaUtils
    {
        private CudaKernel memfillkernel;

        public CudaUtils(FxCuda cuda)
        {

            FxCudaPTX ptxFile = new FxCudaPTX(cuda, "memory", "PTX");
            // init the Cuda functions
            memfillkernel = ptxFile.LoadKernel("memfill");
            ptxFile.Dispose();
        }

        public void MemFill<T>(CudaDeviceVariable<T> Dest,
                            uint offset,
                            CudaDeviceVariable<T> Fill_pattern,
                            uint Fill_len)
        {
            memfillkernel.BlockDimensions = 256;
            memfillkernel.GridDimensions = Fill_len / 256;
            memfillkernel.Run(Dest.DevicePointer,
                Dest.SizeInBytes / 4,
                offset,
                Fill_pattern,
                Fill_pattern.SizeInBytes / 4,
                Fill_len);
        }
    }
}
