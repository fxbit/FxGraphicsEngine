using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using ManagedCuda;
using FxMaths.Vector;
using ManagedCuda.BasicTypes;
using System.Threading;
using System.Runtime.InteropServices;
using FxMaths;

namespace Delaunay
{


    public class BitonicSort<T> where T : struct
    {

        #region Constant Variables

        const uint BITONIC_BLOCK_SIZE = 1024;
        const uint TRANSPOSE_BLOCK_SIZE = 8;

        #endregion



        #region Private Variables

        private int numElements = 0;
        private int MaxNumElements = 0;

        private CudaDeviceVariable<T> d_Input;
        private CudaDeviceVariable<T> d_Output;

        private CudaKernel BitonicSortKernel;
        private CudaKernel MatrixTransposeKernel;

        private FxCuda cuda;
        #endregion



        #region Constructor

        public BitonicSort(FxCuda cuda)
        {
            // link the internal cuda with the external
            this.cuda = cuda;

            FxCudaPTX ptxFile = new FxCudaPTX(cuda, "BitonicSort", "PTX");
            // init the Cuda functions
            BitonicSortKernel = ptxFile.LoadKernel("BitonicSort");
            MatrixTransposeKernel = ptxFile.LoadKernel("MatrixTranspose");
            ptxFile.Dispose();
        } 
        #endregion


        #region Sort

        public void Sort(Boolean Ascending, uint element)
        {
            uint MATRIX_WIDTH = BITONIC_BLOCK_SIZE;
            uint MATRIX_HEIGHT = (uint)(numElements / BITONIC_BLOCK_SIZE);
            int bitonicKernel = (int)(numElements / BITONIC_BLOCK_SIZE);

            int transpose_width_kernel_num = (int)(MATRIX_WIDTH / TRANSPOSE_BLOCK_SIZE);
            int transpose_height_kernel_num = (int)(MATRIX_HEIGHT / TRANSPOSE_BLOCK_SIZE);

            BitonicSettings settings = new BitonicSettings();

            // Sort the data
            // First sort the rows for the levels <= to the block size
            for (uint level = 2; level <= BITONIC_BLOCK_SIZE; level = level * 2)
            {
                settings.g_iLevel = level;
                settings.g_iLevelMask = level;
                settings.g_iWidth = MATRIX_HEIGHT;
                settings.g_iHeight = MATRIX_WIDTH;
                settings.g_iField = element;

                // Sort the row data
                BitonicSortKernel.GridDimensions = bitonicKernel;
                BitonicSortKernel.BlockDimensions = BITONIC_BLOCK_SIZE;
                BitonicSortKernel.Run(settings, d_Input.DevicePointer);
            }

            // Then sort the rows and columns for the levels > than the block size
            // Transpose. Sort the Columns. Transpose. Sort the Rows.
            for (uint level = (BITONIC_BLOCK_SIZE * 2); level <= numElements; level = level * 2)
            {
                settings.g_iLevel = (level / BITONIC_BLOCK_SIZE);
                settings.g_iLevelMask = (uint)((level & ~numElements) / BITONIC_BLOCK_SIZE);
                settings.g_iWidth = MATRIX_WIDTH;
                settings.g_iHeight = MATRIX_HEIGHT;
                settings.g_iField = element;

                // Transpose the data from buffer 1 into buffer 2
                MatrixTransposeKernel.GridDimensions = new ManagedCuda.VectorTypes.dim3(transpose_width_kernel_num, transpose_height_kernel_num);
                MatrixTransposeKernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(TRANSPOSE_BLOCK_SIZE, TRANSPOSE_BLOCK_SIZE);
                MatrixTransposeKernel.Run(settings,
                                          d_Input.DevicePointer,
                                          d_Output.DevicePointer);

                // Sort the transposed column data
                BitonicSortKernel.GridDimensions = bitonicKernel;
                BitonicSortKernel.BlockDimensions = BITONIC_BLOCK_SIZE;
                BitonicSortKernel.Run(settings, 
                                      d_Output.DevicePointer);

                settings.g_iLevel = BITONIC_BLOCK_SIZE;
                settings.g_iLevelMask = level;
                settings.g_iWidth = MATRIX_HEIGHT;
                settings.g_iHeight = MATRIX_WIDTH;
                settings.g_iField = element;

                // Transpose the data from buffer 2 back into buffer 1
                MatrixTransposeKernel.GridDimensions = new ManagedCuda.VectorTypes.dim3(transpose_height_kernel_num, transpose_width_kernel_num);
                MatrixTransposeKernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(TRANSPOSE_BLOCK_SIZE, TRANSPOSE_BLOCK_SIZE);
                MatrixTransposeKernel.Run(settings,
                                          d_Output.DevicePointer,
                                          d_Input.DevicePointer);

                // Sort the row data
                BitonicSortKernel.GridDimensions = bitonicKernel;
                BitonicSortKernel.BlockDimensions = BITONIC_BLOCK_SIZE;
                BitonicSortKernel.Run(settings, d_Input.DevicePointer);
            }

            //TimeStatistics.ClockLap("GPU Sort");
        }

        #endregion

        private void SetNumElements(int num)
        {
            this.numElements = (int)(Math.Pow(2, Math.Ceiling(Math.Log(num, 2))));
            this.numElements = (int)((this.numElements < TRANSPOSE_BLOCK_SIZE * BITONIC_BLOCK_SIZE) ? (int)(TRANSPOSE_BLOCK_SIZE * BITONIC_BLOCK_SIZE) : (int)this.numElements);
        }

        #region Set/Get Data/Results

        public T[] GetResults()
        {
            // copy the results back
            return d_Output;
        }

        public void GetResults(CudaDeviceVariable<T> out_data, SizeT offset, int dataLen, uint primSize)
        {
            // check if the memory that we want to copy exist to the internal data
            if (numElements >= dataLen)
            {
                out_data.CopyToDevice(d_Output.DevicePointer, 0, offset, dataLen * primSize);
            }
        }

        public void GetResults(CudaDeviceVariable<T> out_data, SizeT offsetSrc, SizeT offsetDst, int dataLen, uint primSize)
        {
            // check if the memory that we want to copy exist to the internal data
            if (numElements >= dataLen + offsetSrc)
            {
                out_data.CopyToDevice(d_Output.DevicePointer, offsetSrc, offsetDst, dataLen * primSize);
            }
        }

        public void SetData(CudaDeviceVariable<T> in_data, SizeT offset, int dataLen, uint primSize)
        {

            // calculate the next correct size
            SetNumElements(dataLen);

            // check if we can use the internal memory for the sorting
            // if not reset the internal memory to be able
            if (this.numElements > this.MaxNumElements)
                Prepare(dataLen);

            // fill the data with max value
            this.d_Input.Memset(uint.MaxValue);

            // copy the external data to the internal one
            this.d_Input.CopyToDevice(in_data.DevicePointer, offset, 0, dataLen * primSize);

        } 

        #endregion




        #region Prepare phase

        /// <summary>
        /// Prepare the internal state with the max 
        /// size of internal variables
        /// </summary>
        /// <param name="dataLen"></param>
        /// <param name="MaxMinT"></param>
        public void Prepare(int dataLen)
        {
            // remoeve any previus memory that we have allocate
            DisposeMemory();

            // store the number of elements
            SetNumElements(dataLen);
            this.MaxNumElements = this.numElements;

            // allocate the hw variables
            d_Input = new CudaDeviceVariable<T>(numElements);
            d_Output = new CudaDeviceVariable<T>(numElements);

        }

        #endregion



        #region Dispose

        /// <summary>
        /// Clean all the internal memorys that we 
        /// have allocate from the HW
        /// </summary>
        public void DisposeMemory()
        {
            if (d_Input != null)
                d_Input.Dispose();
            if (d_Output != null)
                d_Output.Dispose();

            d_Input = null;
            d_Output = null;
        }

        /// <summary>
        /// Clean all internal memory and code
        /// </summary>
        public void Dispose()
        {
            // dispose memory
            DisposeMemory();
        } 
        #endregion

    }

    [StructLayout(LayoutKind.Sequential)]
    struct BitonicSettings
    {
        public uint g_iLevel;
        public uint g_iLevelMask;
        public uint g_iWidth;
        public uint g_iHeight;
        public uint g_iField;
    }
}
