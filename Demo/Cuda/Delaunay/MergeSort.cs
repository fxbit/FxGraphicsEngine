using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using ManagedCuda;
using FxMaths.Vector;

namespace Delaunay
{
    public class MergeSort<T> where T : struct
    {
        const uint MAX_SAMPLE_COUNT = 32768;
        const uint SHARED_SIZE_LIMIT = 1024;
        const uint SAMPLE_STRIDE = 128;

        /// <summary>
        /// Internal pointers to the lists that we 
        /// want to sort
        /// </summary>
        private T[] h_list;
        private CudaDeviceVariable<T> d_list;

        private int numElements = 0;
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

        public MergeSort(FxCuda cuda, T[] h_list, CudaDeviceVariable<T> d_list)
        {
            // link the external list with the internals 
            this.h_list = h_list;
            this.d_list = d_list;

            // init the support variables
            uint[] dummy_rank = new uint[MAX_SAMPLE_COUNT];
            d_RanksA = dummy_rank;
            d_RanksB = dummy_rank;
            d_LimitsA = dummy_rank;
            d_LimitsB = dummy_rank;

            // store the number of elements
            numElements = h_list.Length;

            // init the big buffers
            uint[] h_SrcVal = new uint[numElements];
            for (uint i = 0; i < numElements; i++)
                h_SrcVal[i] = i;
            d_SrcVal = h_SrcVal;

            d_DstKey = new CudaDeviceVariable<T>(numElements);
            d_DstVal = new CudaDeviceVariable<uint>(numElements);
            d_BufKey = new CudaDeviceVariable<T>(numElements);
            d_BufVal = new CudaDeviceVariable<uint>(numElements);

            // init the Cuda functions
            mergeSortSharedKernelUp = cuda.LoadPTX("MergeSort", "PTX", "mergeSortSharedKernelUp");
            mergeSortSharedKernelDown = cuda.LoadPTX("MergeSort", "PTX", "mergeSortSharedKernelDown");

            generateSampleRanksKernelUp = cuda.LoadPTX("MergeSort", "PTX", "generateSampleRanksKernelUp");
            generateSampleRanksKernelDown = cuda.LoadPTX("MergeSort", "PTX", "generateSampleRanksKernelDown");

            mergeRanksAndIndicesKernel = cuda.LoadPTX("MergeSort", "PTX", "mergeRanksAndIndicesKernel");

            mergeElementaryIntervalsKernelUp = cuda.LoadPTX("MergeSort", "PTX", "mergeElementaryIntervalsKernelUp");
            mergeElementaryIntervalsKernelDown = cuda.LoadPTX("MergeSort", "PTX", "mergeElementaryIntervalsKernelDown");

        }

        private void mergeSortShared(CudaDeviceVariable<T> d_DstKey,
                                     CudaDeviceVariable<uint> d_DstVal,
                                     CudaDeviceVariable<T> d_SrcKey,
                                     CudaDeviceVariable<uint> d_SrcVal,
                                     Boolean Ascending)
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
                    arrayLength);
            }
            else
            {
                mergeSortSharedKernelDown.BlockDimensions = threadCount;
                mergeSortSharedKernelDown.GridDimensions = blockCount;
                mergeSortSharedKernelDown.Run(d_DstKey.DevicePointer,
                    d_DstVal.DevicePointer,
                    d_SrcKey.DevicePointer,
                    d_SrcVal.DevicePointer,
                    arrayLength);
            }
        }



        private uint iDivUp(uint a, uint b)
        {
            return ((a % b) == 0) ? (a / b) : (a / b + 1);
        }






        private void generateSampleRanks(CudaDeviceVariable<uint> d_RanksA,
                                         CudaDeviceVariable<uint> d_RanksB,
                                         CudaDeviceVariable<T> d_SrcKey,
                                         uint stride,
                                         Boolean Ascending)
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
                    stride, numElements, threadCount);
            }
            else
            {
                generateSampleRanksKernelDown.GridDimensions = iDivUp(threadCount, 256);
                generateSampleRanksKernelDown.BlockDimensions = 256;
                generateSampleRanksKernelDown.Run(d_RanksA.DevicePointer,
                    d_RanksB.DevicePointer,
                    d_SrcKey.DevicePointer,
                    stride, numElements, threadCount);
            }
        }







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

            mergeRanksAndIndicesKernel.Run(
                d_LimitsB.DevicePointer,
                d_RanksB.DevicePointer,
                stride,
                numElements,
                threadCount
            );
        }





        private void mergeElementaryIntervals(CudaDeviceVariable<T> d_DstKey,
                                              CudaDeviceVariable<uint> d_DstVal,
                                              CudaDeviceVariable<T> d_SrcKey,
                                              CudaDeviceVariable<uint> d_SrcVal,
                                              CudaDeviceVariable<uint> d_LimitsA,
                                              CudaDeviceVariable<uint> d_LimitsB,
                                              uint stride,
                                              Boolean Ascending)
        {
            uint lastSegmentElements = (uint)(numElements % (2 * stride));
            uint mergePairs = (uint)((lastSegmentElements > stride) ?
                iDivUp((uint)numElements, SAMPLE_STRIDE) :
                (numElements - lastSegmentElements) / SAMPLE_STRIDE);

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
                    numElements
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
                    numElements
                );
            }
        }










        public void Sort(Boolean Ascending)
        {
            uint stageCount = 0;
            CudaDeviceVariable<T> ikey, okey;
            CudaDeviceVariable<uint> ival, oval;
            CudaDeviceVariable<T> t;
            CudaDeviceVariable<uint> v;

            // find  if is odd or even
            for (uint stride = SHARED_SIZE_LIMIT; stride < numElements; stride <<= 1, stageCount++) ;
            if (stageCount % 2 == 1)
            {
                ikey = d_BufKey;
                ival = d_BufVal;
                okey = d_DstKey;
                oval = d_DstVal;
            }
            else
            {
                ikey = d_DstKey;
                ival = d_DstVal;
                okey = d_BufKey;
                oval = d_BufVal;
            }

            mergeSortShared(ikey, ival, d_list, d_SrcVal, Ascending);

           
            for (uint stride = SHARED_SIZE_LIMIT; stride < numElements; stride <<= 1)
            {
                uint lastSegmentElements = (uint)(numElements % (2 * stride));

                //Find sample ranks and prepare for limiters merge
                generateSampleRanks(d_RanksA, d_RanksB, ikey, stride, Ascending);

                //Merge ranks and indices
                mergeRanksAndIndices(d_LimitsA, d_LimitsB, d_RanksA, d_RanksB, stride);

                //Merge elementary intervals
                mergeElementaryIntervals(okey, oval, ikey, ival, d_LimitsA, d_LimitsB, stride, Ascending);

                if (lastSegmentElements <= stride)
                {
                    //Last merge segment consists of a single array which just needs to be passed through
                    okey.CopyToDevice(ikey.DevicePointer, (numElements - lastSegmentElements), (numElements - lastSegmentElements), lastSegmentElements * 4 * 2);
                    oval.CopyToDevice(ival.DevicePointer, (numElements - lastSegmentElements), (numElements - lastSegmentElements), lastSegmentElements * 4);
                }

                
                t = ikey;
                ikey = okey;
                okey = t;

                v = ival;
                ival = oval;
                oval = v;
            }

            // copy the results back
            h_list = d_DstKey;
        }

        public T[] GetResults()
        {
            return h_list;
        }
    }
}
