using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using ManagedCuda;
using FxMaths.Vector;
using ManagedCuda.BasicTypes;
using System.Threading;

namespace Delaunay
{
    public class MergeSort<T> where T : struct
    {

        #region Constant Variables

        const uint MAX_SAMPLE_COUNT = 32768;
        const uint SHARED_SIZE_LIMIT = 1024;
        const uint SAMPLE_STRIDE = 128;

        #endregion



        #region Private Variables

        /// <summary>
        /// Internal pointers to the lists that we 
        /// want to sort
        /// </summary>
        private CudaDeviceVariable<T> d_SrcKey;

        private int numElements = 0;
        private int MaxNumElements = 0;
        private CudaDeviceVariable<uint> d_RanksA, d_RanksB;
        private CudaDeviceVariable<uint> d_LimitsA, d_LimitsB;
        private CudaDeviceVariable<T> d_BufKey, d_DstKey;
        private CudaDeviceVariable<uint> d_SrcVal, d_BufVal, d_DstVal;

        private CudaKernel mergeSortSharedKernelUp;
        private CudaKernel mergeSortSharedKernelDown;
        private CudaKernel generateSampleRanksKernelUp;
        private CudaKernel generateSampleRanksKernelDown;
        private CudaKernel mergeRanksAndIndicesKernel;
        private CudaKernel mergeElementaryIntervalsKernelUp;
        private CudaKernel mergeElementaryIntervalsKernelDown;

        private uint[] h_SrcVal;

        private FxCuda cuda;
        #endregion



        #region Constructor

        public MergeSort(FxCuda cuda)
        {
            // link the internal cuda with the external
            this.cuda = cuda;

            // init the support variables
            uint[] dummy_rank = new uint[MAX_SAMPLE_COUNT];
            d_RanksA = dummy_rank;
            d_RanksB = dummy_rank;
            d_LimitsA = dummy_rank;
            d_LimitsB = dummy_rank;


            FxCudaPTX ptxFile = new FxCudaPTX(cuda, "MergeSort", "PTX");
            // init the Cuda functions
            mergeSortSharedKernelUp = ptxFile.LoadKernel("mergeSortSharedKernelUp");
            mergeSortSharedKernelDown = ptxFile.LoadKernel("mergeSortSharedKernelDown");

            generateSampleRanksKernelUp = ptxFile.LoadKernel("generateSampleRanksKernelUp");
            generateSampleRanksKernelDown = ptxFile.LoadKernel("generateSampleRanksKernelDown");

            mergeRanksAndIndicesKernel = ptxFile.LoadKernel("mergeRanksAndIndicesKernel");

            mergeElementaryIntervalsKernelUp = ptxFile.LoadKernel("mergeElementaryIntervalsKernelUp");
            mergeElementaryIntervalsKernelDown = ptxFile.LoadKernel("mergeElementaryIntervalsKernelDown");

            ptxFile.Dispose();
        } 
        #endregion



        #region mergeSortShared

        private void mergeSortShared(CudaDeviceVariable<T> d_DstKey,
                                     CudaDeviceVariable<uint> d_DstVal,
                                     CudaDeviceVariable<T> d_SrcKey,
                                     CudaDeviceVariable<uint> d_SrcVal,
                                     Boolean Ascending,
                                     uint element)
        {
            uint arrayLength = SHARED_SIZE_LIMIT;
            int batchSize = (int)(numElements / SHARED_SIZE_LIMIT);
            uint blockCount = (uint)(batchSize * arrayLength / SHARED_SIZE_LIMIT);
            uint threadCount = SHARED_SIZE_LIMIT / 2;


            if (Ascending)
            {
                mergeSortSharedKernelUp.BlockDimensions = threadCount;
                mergeSortSharedKernelUp.GridDimensions = blockCount;
                mergeSortSharedKernelUp.Run(d_DstKey.DevicePointer,
                    d_DstVal.DevicePointer,
                    d_SrcKey.DevicePointer,
                    d_SrcVal.DevicePointer,
                    arrayLength,
                    element);
            }
            else
            {
                mergeSortSharedKernelDown.BlockDimensions = threadCount;
                mergeSortSharedKernelDown.GridDimensions = blockCount;
                mergeSortSharedKernelDown.Run(d_DstKey.DevicePointer,
                    d_DstVal.DevicePointer,
                    d_SrcKey.DevicePointer,
                    d_SrcVal.DevicePointer,
                    arrayLength,
                    element);
            }
        } 

        #endregion



        #region Utilities

        private uint iDivUp(uint a, uint b)
        {
            return ((a % b) == 0) ? (a / b) : (a / b + 1);
        }

        private uint getSampleCount(uint dividend)
        {
            return iDivUp(dividend, SAMPLE_STRIDE);
        }

        #endregion



        #region generateSampleRanks

        private void generateSampleRanks(CudaDeviceVariable<uint> d_RanksA,
                                         CudaDeviceVariable<uint> d_RanksB,
                                         CudaDeviceVariable<T> d_SrcKey,
                                         uint stride,
                                         Boolean Ascending,
                                         uint element)
        {
            uint lastSegmentElements = (uint)(numElements % (2 * stride));
            uint threadCount = (uint)((lastSegmentElements > stride) ?
                (numElements + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) :
                (numElements - lastSegmentElements) / (2 * SAMPLE_STRIDE));

            if (Ascending)
            {
                generateSampleRanksKernelUp.GridDimensions = iDivUp(threadCount, 256);
                generateSampleRanksKernelUp.BlockDimensions = 256;
                generateSampleRanksKernelUp.Run(d_RanksA.DevicePointer,
                    d_RanksB.DevicePointer,
                    d_SrcKey.DevicePointer,
                    stride, numElements, threadCount, element);
            }
            else
            {
                generateSampleRanksKernelDown.GridDimensions = iDivUp(threadCount, 256);
                generateSampleRanksKernelDown.BlockDimensions = 256;
                generateSampleRanksKernelDown.Run(d_RanksA.DevicePointer,
                    d_RanksB.DevicePointer,
                    d_SrcKey.DevicePointer,
                    stride, numElements, threadCount, element);
            }
        }
        
        #endregion



        #region mergeRanksAndIndices

        private void mergeRanksAndIndices(CudaDeviceVariable<uint> d_LimitsA,
                                          CudaDeviceVariable<uint> d_LimitsB,
                                          CudaDeviceVariable<uint> d_RanksA,
                                          CudaDeviceVariable<uint> d_RanksB,
                                          uint stride)
        {
            uint lastSegmentElements = (uint)(numElements % (2 * stride));
            uint threadCount = (uint)((lastSegmentElements > stride) ?
                (numElements + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) :
                (numElements - lastSegmentElements) / (2 * SAMPLE_STRIDE));

            mergeRanksAndIndicesKernel.GridDimensions = iDivUp(threadCount, 256);
            mergeRanksAndIndicesKernel.BlockDimensions = 256;
            mergeRanksAndIndicesKernel.Run(
                d_LimitsA.DevicePointer,
                d_RanksA.DevicePointer,
                stride,
                numElements,
                threadCount
            );

            cuda.ctx.Synchronize();

            mergeRanksAndIndicesKernel.GridDimensions = iDivUp(threadCount, 256);
            mergeRanksAndIndicesKernel.BlockDimensions = 256;
            mergeRanksAndIndicesKernel.Run(
                d_LimitsB.DevicePointer,
                d_RanksB.DevicePointer,
                stride,
                numElements,
                threadCount
            );
        } 

        #endregion




        #region mergeElementaryIntervals

        private void mergeElementaryIntervals(CudaDeviceVariable<T> d_DstKey,
                                              CudaDeviceVariable<uint> d_DstVal,
                                              CudaDeviceVariable<T> d_SrcKey,
                                              CudaDeviceVariable<uint> d_SrcVal,
                                              CudaDeviceVariable<uint> d_LimitsA,
                                              CudaDeviceVariable<uint> d_LimitsB,
                                              uint stride,
                                              Boolean Ascending,
                                              uint element)
        {
            uint lastSegmentElements = (uint)(numElements % (2 * stride));
            uint mergePairs = (uint)((lastSegmentElements > stride) ?
                getSampleCount((uint)numElements) :
                (numElements - lastSegmentElements) / SAMPLE_STRIDE);

            cuda.ctx.Synchronize();

            if (Ascending)
            {
                mergeElementaryIntervalsKernelUp.BlockDimensions = SAMPLE_STRIDE;
                mergeElementaryIntervalsKernelUp.GridDimensions = mergePairs;
                mergeElementaryIntervalsKernelUp.Run(
                    d_DstKey.DevicePointer,
                    d_DstVal.DevicePointer,
                    d_SrcKey.DevicePointer,
                    d_SrcVal.DevicePointer,
                    d_LimitsA.DevicePointer,
                    d_LimitsB.DevicePointer,
                    stride,
                    numElements,
                    element
                );
            }
            else
            {
                mergeElementaryIntervalsKernelDown.BlockDimensions = SAMPLE_STRIDE;
                mergeElementaryIntervalsKernelDown.GridDimensions = mergePairs;
                mergeElementaryIntervalsKernelDown.Run(
                    d_DstKey.DevicePointer,
                    d_DstVal.DevicePointer,
                    d_SrcKey.DevicePointer,
                    d_SrcVal.DevicePointer,
                    d_LimitsA.DevicePointer,
                    d_LimitsB.DevicePointer,
                    stride,
                    numElements,
                    element
                );
            }
        } 
        #endregion



        #region Sort

        public void Sort(Boolean Ascending, uint element)
        {
            CudaDeviceVariable<T> ikey, okey;
            CudaDeviceVariable<uint> ival, oval;
            CudaDeviceVariable<T> t;
            CudaDeviceVariable<uint> v;

            ikey = d_BufKey;
            ival = d_BufVal;
            okey = d_DstKey;
            oval = d_DstVal;

            mergeSortShared(ikey, ival, d_SrcKey, d_SrcVal, Ascending, element);

            for (uint stride = SHARED_SIZE_LIMIT; stride < numElements; stride <<= 1)
            {
                uint lastSegmentElements = (uint)(numElements % (2 * stride));

                //Find sample ranks and prepare for limiters merge
                generateSampleRanks(d_RanksA, d_RanksB, ikey, stride, Ascending, element);

                //Merge ranks and indices
                mergeRanksAndIndices(d_LimitsA, d_LimitsB, d_RanksA, d_RanksB, stride);

                //Merge elementary intervals
                mergeElementaryIntervals(okey, oval, ikey, ival, d_LimitsA, d_LimitsB, stride, Ascending, element);

                if (lastSegmentElements <= stride)
                {
                    //Last merge segment consists of a single array which just needs to be passed through
                    okey.CopyToDevice(ikey.DevicePointer, (numElements - lastSegmentElements), (numElements - lastSegmentElements), lastSegmentElements * ikey.TypeSize);
                    oval.CopyToDevice(ival.DevicePointer, (numElements - lastSegmentElements), (numElements - lastSegmentElements), lastSegmentElements * ival.TypeSize);
                }

#if false
                // swap
                okey = Interlocked.Exchange<CudaDeviceVariable<T>>(ref ikey, okey);
                oval = Interlocked.Exchange<CudaDeviceVariable<uint>>(ref ival, oval);
#else
                t = ikey;
                ikey = okey;
                okey = t;
                v = ival;
                ival = oval;
                oval = v;
#endif

            }
        }

        #endregion



        #region Set/Get Data/Results

        public T[] GetResults()
        {
            // copy the results back
            return d_DstKey;
        }

        public void GetResults(CudaDeviceVariable<T> out_data, SizeT offset, int dataLen, uint primSize)
        {
            // check if the memory that we want to copy exist to the internal data
            if (numElements >= dataLen)
            {
                out_data.CopyToDevice(d_DstKey.DevicePointer, 0, offset, dataLen * primSize);
            }
        }

        public void GetResults(CudaDeviceVariable<T> out_data, SizeT offsetSrc, SizeT offsetDst, int dataLen, uint primSize)
        {
            // check if the memory that we want to copy exist to the internal data
            if (numElements >= dataLen + offsetSrc)
            {
                out_data.CopyToDevice(d_DstKey.DevicePointer, offsetSrc, offsetDst, dataLen * primSize);
            }
        }

        public void SetData(CudaDeviceVariable<T> in_data, SizeT offset, int dataLen, uint primSize)
        {

            // calculate the next correct size
            this.numElements = (int)(Math.Pow(2, Math.Ceiling(Math.Log(dataLen, 2))));

            // check if we can use the internal memory for the sorting
            // if not reset the internal memory to be able
            if (this.numElements > this.MaxNumElements)
                Prepare(dataLen);

            // fill the data with max value
            this.d_SrcKey.Memset(uint.MaxValue);

            // copy the external data to the internal one
            this.d_SrcKey.CopyToDevice(in_data.DevicePointer, offset, 0, dataLen * primSize);

            // reset the d_srcVal
            d_SrcVal.CopyToDevice(h_SrcVal, 0, 0, numElements * d_SrcVal.TypeSize);
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
            this.numElements = (int)(Math.Pow(2, Math.Ceiling(Math.Log(dataLen, 2))));
            this.MaxNumElements = this.numElements;

            // allocate the hw variables
            d_SrcKey = new CudaDeviceVariable<T>(numElements);
            d_DstKey = new CudaDeviceVariable<T>(numElements);
            d_DstVal = new CudaDeviceVariable<uint>(numElements);
            d_BufKey = new CudaDeviceVariable<T>(numElements);
            d_BufVal = new CudaDeviceVariable<uint>(numElements);
            d_SrcVal = new CudaDeviceVariable<uint>(numElements);

            h_SrcVal = new uint[numElements];
            for (uint i = 0; i < numElements; i++)
                h_SrcVal[i] = i;
        }

        #endregion



        #region Dispose

        /// <summary>
        /// Clean all the internal memorys that we 
        /// have allocate from the HW
        /// </summary>
        public void DisposeMemory()
        {
            if (d_SrcKey != null)
                d_SrcKey.Dispose();
            if (d_DstKey != null)
                d_DstKey.Dispose();
            if (d_DstVal != null)
                d_DstVal.Dispose();
            if (d_BufKey != null)
                d_BufKey.Dispose();
            if (d_BufVal != null)
                d_BufVal.Dispose();
            if (d_SrcVal != null)
                d_SrcVal.Dispose();

            d_SrcKey = null;
            d_DstKey = null;
            d_DstVal = null;
            d_BufKey = null;
            d_BufVal = null;
            h_SrcVal = null;
            d_SrcVal = null;
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
}
