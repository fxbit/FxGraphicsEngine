using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using ManagedCuda;
using ManagedCuda.BasicTypes;
using System.Runtime.InteropServices;
using ManagedCuda.VectorTypes;

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
                            int offset,
                            CudaDeviceVariable<T> Fill_pattern,
                            int Fill_len) where T : struct
        {
            // translate the offset to uint steps
            offset = offset * Dest.TypeSize / uint1.SizeOf;
            Fill_len = Fill_len * Fill_pattern.TypeSize / uint1.SizeOf;

            memfillkernel.BlockDimensions = 32;
            memfillkernel.GridDimensions = Fill_len / 32;
            memfillkernel.Run(Dest.DevicePointer,
                Dest.SizeInBytes / uint1.SizeOf,
                offset,
                Fill_pattern.DevicePointer,
                Fill_pattern.SizeInBytes / uint1.SizeOf,
                Fill_len);
        }
    }
}
