using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;


using FxMaths;
using FxMaths.GMaps;
using FxMaths.GUI;
using FxMaths.Vector;

using FXFramework;

using SharpDX;
using SharpDX.DXGI;
using SharpDX.Direct3D11;
using Buffer = SharpDX.Direct3D11.Buffer;
using D3D = SharpDX.Direct3D11;

using ComputeShader = GraphicsEngine.Core.ComputeShader;
using System.IO;
using System.Runtime.InteropServices;

namespace Delaunay
{
    [StructLayout(LayoutKind.Explicit, Size = 32)] //20
    public struct cbBitonic
    {
        [FieldOffset(0)]
        public uint g_iLevel;

        [FieldOffset(4)]
        public uint g_iLevelMask;

        [FieldOffset(8)]
        public uint g_iWidth;

        [FieldOffset(12)]
        public uint g_iHeight;

        [FieldOffset(16)]
        public uint g_iField;
    }

    [StructLayout(LayoutKind.Explicit, Size = 32)] //16
    public struct CB_Split
    {
        [FieldOffset(0)]
        public uint NumElements;

        [FieldOffset(4)]
        public uint NumRegionsH;

        [FieldOffset(8)]
        public uint NumRegionsV;

        [FieldOffset(12)]
        public uint SplitOffset;

        [FieldOffset(16)]
        public uint RegionIndex;

        [FieldOffset(20)]
        public uint SetMax;
    }

    public class BitonicSort
    {

        #region CS Variables

        /// <summary>
        /// Compute Shader for 
        /// </summary>
        ComputeShader CSBitonicSort = null;


        /// <summary>
        /// Compute Shader for 
        /// </summary>
        ComputeShader CSMatrixTranspose = null;

        /// <summary>
        /// Compute Shader for 
        /// </summary>
        ComputeShader CSFindSplitIndexH = null;

        /// <summary>
        /// Compute Shader for 
        /// </summary>
        ComputeShader CSFindSplitIndexV = null;

        /// <summary>
        /// Compute Shader for 
        /// </summary>
        ComputeShader CSFillRegionInfo = null;


        /// <summary>
        /// Compute Shader for 
        /// </summary>
        ComputeShader CSCopyRegion = null;

        /// <summary>
        /// Compute Shader for 
        /// </summary>
        ComputeShader CSCopySubBuffer = null;

        #endregion



        #region Variables

        // The number of elements to sort is limited to an even power of 2
        // At minimum 8,192 elements - BITONIC_BLOCK_SIZE * TRANSPOSE_BLOCK_SIZE
        // At maximum 262,144 elements - BITONIC_BLOCK_SIZE * BITONIC_BLOCK_SIZE
        uint BITONIC_BLOCK_SIZE = 1024;
        uint TRANSPOSE_BLOCK_SIZE = 16;

        Buffer g_pBuffer1;
        Buffer g_pBuffer2;
        Buffer g_pBuffer3;

        Buffer g_pBuffer_HRegions;
        Buffer g_pBuffer_VRegions;

        FxResourceVariableList rvInput;
        FxResourceVariableList rvData;

        UnorderedAccessView uavBuffer1;
        UnorderedAccessView uavBuffer2;
        UnorderedAccessView uavBuffer3;

        FxResourceVariableList rvRegionsInput;
        FxResourceVariableList rvRegionsOutput;

        UnorderedAccessView uavHRegions;
        UnorderedAccessView uavVRegions;

        FXConstantBuffer<cbBitonic> CBBitonic;
        FXConstantBuffer<cbBitonic> CBTranspose;

        Buffer stagingResult;

        FXConstantBuffer<CB_Split> CBFindSplitIndexH;
        FXConstantBuffer<CB_Split> CBFindSplitIndexV;
        FXConstantBuffer<CB_Split> CBFillRegionInfo;
        FXConstantBuffer<CB_Split> CBCopyRegion;
        FXConstantBuffer<CB_Split> CBCopySubBuffer;
        
        #endregion



        #region Private Variables

        /// <summary>
        /// The max number of elements.
        /// This must be power of 2
        /// </summary>
        private int MaxNumElements;

        /// <summary>
        /// The max number of input elements.
        /// </summary>
        private int MaxNumInputElements;


        /// <summary>
        /// Local ref to graphic engine
        /// </summary>
        private SharpDX.Direct3D11.Device dev;
        
        #endregion



        #region Constractors

        /// <summary>
        /// Start the bitonic with max bitonic size
        /// </summary>
        /// <param name="maxNumElements"></param>
        public BitonicSort(int maxNumElements, SharpDX.Direct3D11.Device device)
        {
            // link the device 
            this.dev = device;

            TimeStatistics.StartClock();

            // compile the shaders
            if (false)
            {
                CSBitonicSort = new ComputeShader(@"Delaunay\Shaders\BitonicSort.hlsl", "BitonicSort", @"Delaunay\Shaders\");
                CSMatrixTranspose = new ComputeShader(@"Delaunay\Shaders\BitonicSort.hlsl", "MatrixTranspose", @"Delaunay\Shaders\");
                CSFindSplitIndexH = new ComputeShader(@"Delaunay\Shaders\RegionSplit.hlsl", "FindSplitIndexH", @"Delaunay\Shaders\");
                CSFindSplitIndexV = new ComputeShader(@"Delaunay\Shaders\RegionSplit.hlsl", "FindSplitIndexV", @"Delaunay\Shaders\");
                CSFillRegionInfo = new ComputeShader(@"Delaunay\Shaders\RegionSplit.hlsl", "FillRegionInfo", @"Delaunay\Shaders\");
                CSCopyRegion = new ComputeShader(@"Delaunay\Shaders\RegionSplit.hlsl", "CopyRegion", @"Delaunay\Shaders\");
                CSCopySubBuffer = new ComputeShader(@"Delaunay\Shaders\RegionSplit.hlsl", "CopySubBuffer", @"Delaunay\Shaders\");
            }
            else
            {
                CSBitonicSort = new ComputeShader(@"Delaunay\Shaders_Prebuild\BitonicSort.BitonicSort.fxo");
                CSMatrixTranspose = new ComputeShader(@"Delaunay\Shaders_Prebuild\BitonicSort.MatrixTranspose.fxo");
                CSFindSplitIndexH = new ComputeShader(@"Delaunay\Shaders_Prebuild\RegionSplit.FindSplitIndexH.fxo");
                CSFindSplitIndexV = new ComputeShader(@"Delaunay\Shaders_Prebuild\RegionSplit.FindSplitIndexV.fxo");
                CSFillRegionInfo = new ComputeShader(@"Delaunay\Shaders_Prebuild\RegionSplit.FillRegionInfo.fxo");
                CSCopyRegion = new ComputeShader(@"Delaunay\Shaders_Prebuild\RegionSplit.CopyRegion.fxo");
                CSCopySubBuffer = new ComputeShader(@"Delaunay\Shaders_Prebuild\RegionSplit.CopySubBuffer.fxo");
            }

            TimeStatistics.ClockLap("Load Shaders");

            // set the max num of elements
            MaxNumInputElements = maxNumElements;
            MaxNumElements = (int)Math.Pow(2, Math.Ceiling(Math.Log(MaxNumInputElements, 2)));

            MaxNumElements = (int)((MaxNumElements < BITONIC_BLOCK_SIZE * TRANSPOSE_BLOCK_SIZE) ? (int)BITONIC_BLOCK_SIZE * (int)TRANSPOSE_BLOCK_SIZE : MaxNumElements);
            MaxNumElements = (int)Math.Pow(2, Math.Ceiling(Math.Log(MaxNumElements, 2)));

            // init the buffers
            //InitBuffers();

        } 
        #endregion



        #region Init buffers

        public void InitBuffers(out Buffer Points, out Buffer Regions)
        {
            // get the number of the points
            int SizeOfPoint = ComputeShader.SizeOfFloat2;

            // allocate the 2 buffers
            g_pBuffer1 = ComputeShader.CreateBuffer(MaxNumElements, SizeOfPoint, AccessViewType.SRV | AccessViewType.UAV);
            g_pBuffer1.DebugName = "g_pBuffer1";
            g_pBuffer2 = ComputeShader.CreateBuffer(MaxNumElements, SizeOfPoint, AccessViewType.UAV);
            g_pBuffer2.DebugName = "g_pBuffer2";
            g_pBuffer3 = ComputeShader.CreateBuffer(MaxNumElements, SizeOfPoint, AccessViewType.UAV);
            g_pBuffer3.DebugName = "g_pBuffer3";

            g_pBuffer_HRegions = ComputeShader.CreateBuffer(20000, RegionInfo.GetStructSize(), AccessViewType.UAV);
            g_pBuffer_HRegions.DebugName = "g_pBuffer_HRegions";
            g_pBuffer_VRegions = ComputeShader.CreateBuffer(20000, RegionInfo.GetStructSize(), AccessViewType.UAV);
            g_pBuffer_VRegions.DebugName = "g_pBuffer_VRegions";

            // create the UAVs
            uavBuffer1 = FXResourceVariable.InitUAVResource(this.dev, g_pBuffer1);
            uavBuffer2 = FXResourceVariable.InitUAVResource(this.dev, g_pBuffer2);
            uavBuffer3 = FXResourceVariable.InitUAVResource(this.dev, g_pBuffer3);
            uavHRegions = FXResourceVariable.InitUAVResource(this.dev, g_pBuffer_HRegions);
            uavVRegions = FXResourceVariable.InitUAVResource(this.dev, g_pBuffer_VRegions);

            rvData = new FxResourceVariableList();
            rvInput = new FxResourceVariableList();
            rvRegionsInput = new FxResourceVariableList();
            rvRegionsOutput = new FxResourceVariableList();

            // link the buffers with shaders
            rvData.AddResourceFromShader(CSBitonicSort.m_effect,"Data");

            
            rvData.AddResourceFromShader(CSMatrixTranspose.m_effect, "Data");
            rvData.AddResourceFromShader(CSCopyRegion.m_effect, "Data");
            rvData.AddResourceFromShader(CSCopySubBuffer.m_effect, "Data");

            rvInput.AddResourceFromShader(CSMatrixTranspose.m_effect, "Input");
            rvInput.AddResourceFromShader(CSFindSplitIndexH.m_effect, "Input");
            rvInput.AddResourceFromShader(CSFindSplitIndexV.m_effect, "Input");
            rvInput.AddResourceFromShader(CSCopyRegion.m_effect, "Input");
            rvInput.AddResourceFromShader(CSCopySubBuffer.m_effect, "Input");

            rvRegionsInput.AddResourceFromShader(CSFindSplitIndexH.m_effect, "RegionInfoInput");
            rvRegionsInput.AddResourceFromShader(CSFindSplitIndexV.m_effect, "RegionInfoInput");
            rvRegionsInput.AddResourceFromShader(CSCopyRegion.m_effect, "RegionInfoInput");
            rvRegionsInput.AddResourceFromShader(CSFillRegionInfo.m_effect, "RegionInfoInput");

            rvRegionsOutput.AddResourceFromShader(CSFindSplitIndexH.m_effect, "RegionInfoOutput");
            rvRegionsOutput.AddResourceFromShader(CSFindSplitIndexV.m_effect, "RegionInfoOutput");
            rvRegionsOutput.AddResourceFromShader(CSCopyRegion.m_effect, "RegionInfoOutput");
            rvRegionsOutput.AddResourceFromShader(CSFillRegionInfo.m_effect, "RegionInfoOutput");

            // Bind the CB
            CBBitonic = CSBitonicSort.m_effect.GetConstantBufferByName<cbBitonic>("CB");
            CBTranspose = CSMatrixTranspose.m_effect.GetConstantBufferByName<cbBitonic>("CB");

            CBFindSplitIndexH = CSFindSplitIndexH.m_effect.GetConstantBufferByName<CB_Split>("CB_Split");
            CBFindSplitIndexV = CSFindSplitIndexV.m_effect.GetConstantBufferByName<CB_Split>("CB_Split");
            CBFillRegionInfo = CSFillRegionInfo.m_effect.GetConstantBufferByName<CB_Split>("CB_Split");
            CBCopyRegion =CSCopyRegion.m_effect.GetConstantBufferByName<CB_Split>("CB_Split");
            CBCopySubBuffer = CSCopySubBuffer.m_effect.GetConstantBufferByName<CB_Split>("CB_Split");

            // create a staging buffer
            stagingResult = ComputeShader.CreateStagingBuffer(g_pBuffer1);

            // set the output
            Points = g_pBuffer3;
            Regions = g_pBuffer_HRegions; // this is complex :P
        }
        
        #endregion



        #region Fill Data

        /// <summary>
        /// Fill the buffers with data
        /// </summary>
        /// <param name="pointsList"></param>
        public void FillData(List<IVertex<float>> pointsList)
        {
            FXResourceVariable.WriteBufferVertex<float>(dev, stagingResult, g_pBuffer3, pointsList);
        }
        
        #endregion



        #region Results

        /// <summary>
        /// Get result
        /// </summary>
        /// <returns></returns>
        public List<IVertex<float>> GetResult()
        {
            // Download the data
            float[] GPUSortResult = new float[g_pBuffer1.Description.SizeInBytes / ComputeShader.SizeOfFloat1];
            FXResourceVariable.ReadBuffer<float>(this.dev, stagingResult, g_pBuffer3, ref GPUSortResult);

            TimeStatistics.ClockLap("GPU Read");

            // create the list 
            List<IVertex<float>> vecList = new List<IVertex<float>>();
            for (int i = 0; i < GPUSortResult.Length; i += 2)
            {
                vecList.Add(new FxVector2f(GPUSortResult[i], GPUSortResult[i + 1]));
            }

            return vecList;
        }

        
        #endregion



        #region GPU sorting

        void SetConstants(uint iLevel, uint iLevelMask, uint iWidth, uint iHeight, uint iField)
        {
            cbBitonic local;

            local.g_iLevel = iLevel;
            local.g_iLevelMask = iLevelMask;
            local.g_iWidth = iWidth;
            local.g_iHeight = iHeight;
            local.g_iField = iField;

            CBBitonic.UpdateValue(local);
            CBTranspose.UpdateValue(local);
        }

        public void Execute(uint field, uint NUM_ELEMENTS)
        {

            uint MATRIX_WIDTH = BITONIC_BLOCK_SIZE;
            uint MATRIX_HEIGHT = NUM_ELEMENTS / BITONIC_BLOCK_SIZE;
            int bitonicKernel = (int)(NUM_ELEMENTS / BITONIC_BLOCK_SIZE);

            int transpose_width_kernel_num = (int)(MATRIX_WIDTH / TRANSPOSE_BLOCK_SIZE);
            int transpose_height_kernel_num = (int)(MATRIX_HEIGHT / TRANSPOSE_BLOCK_SIZE);

            
            // set the input and the data
            rvData.SetViewsToResources(uavBuffer1);

            // Sort the data
            // First sort the rows for the levels <= to the block size
            for (uint level = 2; level <= BITONIC_BLOCK_SIZE; level = level * 2)
            {
                SetConstants(level, level, MATRIX_HEIGHT, MATRIX_WIDTH, field);

                // Sort the row data
                CSBitonicSort.Execute(bitonicKernel, 1);
            }

            // Then sort the rows and columns for the levels > than the block size
            // Transpose. Sort the Columns. Transpose. Sort the Rows.
            for (uint level = (BITONIC_BLOCK_SIZE * 2); level <= NUM_ELEMENTS; level = level * 2)
            {

                SetConstants((level / BITONIC_BLOCK_SIZE), (level & ~NUM_ELEMENTS) / BITONIC_BLOCK_SIZE, MATRIX_WIDTH, MATRIX_HEIGHT, field);

                // Transpose the data from buffer 1 into buffer 2
                rvData.SetViewsToResources(uavBuffer2);
                rvInput.SetViewsToResources(uavBuffer1);
                CSMatrixTranspose.Execute(transpose_width_kernel_num, transpose_height_kernel_num);

                // Sort the transposed column data
                CSBitonicSort.Execute(bitonicKernel, 1);

                SetConstants(BITONIC_BLOCK_SIZE, level, MATRIX_HEIGHT, MATRIX_WIDTH, field);

                // Transpose the data from buffer 2 back into buffer 1
                rvData.SetViewsToResources(uavBuffer1);
                rvInput.SetViewsToResources(uavBuffer2);
                CSMatrixTranspose.Execute(transpose_height_kernel_num, transpose_width_kernel_num);

                // Sort the row data
                CSBitonicSort.Execute(bitonicKernel, 1);

            }

            //TimeStatistics.ClockLap("GPU Sort");
        }

        #endregion



        #region Split

        public void Split(int num)
        {
            // find the split points 
            int PointsPerRegion = num;
            int RegionsNum = (int)Math.Ceiling((float)MaxNumInputElements / (float)PointsPerRegion);

            int HorizontalRegions = (int)Math.Floor(Math.Sqrt(RegionsNum));
            int HorizontalRegionsOffset = (int)Math.Floor((float)MaxNumInputElements / (float)HorizontalRegions);
            int VerticalRegions = (int)Math.Floor((float)RegionsNum / (float)HorizontalRegions);
            int VerticalRegionsOffset = (int)Math.Floor((float)HorizontalRegionsOffset / (float)VerticalRegions);

            float threadNum = 128.0f;
            float threadNum_V_Y = 8.0f;
            float threadCopyNum = 1024.0f;

            // recalc the region number
            RegionsNum = HorizontalRegions * VerticalRegions;


            // copy the horizontal region to the sort buffer
            rvInput.SetViewsToResources(uavBuffer3);
            rvData.SetViewsToResources(uavBuffer1);

            {
                // update the CB
                CB_Split localCB = new CB_Split();
                localCB.NumElements = (uint)MaxNumElements;
                localCB.NumRegionsH = (uint)MaxNumInputElements;
                localCB.SplitOffset = 0;
                localCB.SetMax = (uint)1; // set that we set the remain data to max
                CBCopySubBuffer.UpdateValue(localCB);
            }

            // execute the copy
            CSCopySubBuffer.Execute((int)Math.Ceiling((float)MaxNumElements / threadCopyNum), 1);

            // sort base on X
            this.Execute(1, (uint)MaxNumElements);


            // find the split points
            {
                rvInput.SetViewsToResources(uavBuffer1);

                // set the output region
                rvRegionsOutput.SetViewsToResources(uavHRegions);

                // update the CB
                CB_Split localCB = new CB_Split();
                localCB.NumElements = (uint)MaxNumInputElements;
                localCB.NumRegionsH = (uint)HorizontalRegions;
                localCB.SplitOffset = (uint)HorizontalRegionsOffset;
                CBFindSplitIndexH.UpdateValue(localCB);

                // execute the splitting
                CSFindSplitIndexH.Execute((int)Math.Ceiling((float)HorizontalRegions / threadNum), 1);

                rvRegionsInput.SetViewsToResources(uavHRegions);
                rvRegionsOutput.SetViewsToResources(uavVRegions);

                CBFillRegionInfo.UpdateValue(localCB);

                // fill the region info
                CSFillRegionInfo.Execute((int)Math.Ceiling((float)HorizontalRegions / threadNum), 1);

            }

            // copy the correct result to extra buffer3
            dev.ImmediateContext.CopyResource(g_pBuffer1, g_pBuffer3);
            dev.ImmediateContext.CopyResource(g_pBuffer_VRegions, g_pBuffer_HRegions);

            // sort SubRegions
            {

                // set the region input resources
                rvRegionsInput.SetViewsToResources(uavHRegions);
                rvRegionsOutput.SetViewsToResources(uavVRegions);

                for (int i = 0; i < HorizontalRegions; i++)
                {

                    uint NumH = (uint)Math.Pow(2, Math.Ceiling(Math.Log(HorizontalRegionsOffset, 2)));
                    NumH = (NumH < BITONIC_BLOCK_SIZE * TRANSPOSE_BLOCK_SIZE) ? BITONIC_BLOCK_SIZE * TRANSPOSE_BLOCK_SIZE : NumH;

                    // copy the horizontal region to the sort buffer
                    rvInput.SetViewsToResources(uavBuffer3);
                    rvData.SetViewsToResources(uavBuffer1);

                    {
                        // update the CB
                        CB_Split localCB = new CB_Split();
                        localCB.NumElements = (uint)NumH;
                        localCB.RegionIndex = (uint)i;
                        localCB.SetMax = (uint)1; // set that we set the remain data to max
                        CBCopyRegion.UpdateValue(localCB);
                    }

                    // execute the copy
                    CSCopyRegion.Execute((int)Math.Ceiling((float)NumH / threadCopyNum), 1);

                    // start the sorting of the new buffer
                    this.Execute(0, NumH);

                    // copy the result back to buffer3
                    rvInput.SetViewsToResources(uavBuffer1);
                    rvData.SetViewsToResources(uavBuffer3);

                    {
                        // update the CB
                        CB_Split localCB = new CB_Split();
                        localCB.NumElements = (uint)NumH;
                        localCB.RegionIndex = (uint)i;
                        localCB.SetMax = 0; // set that we set the remain data to max
                        CBCopyRegion.UpdateValue(localCB);
                    }

                    // execute the copy
                    CSCopyRegion.Execute((int)Math.Ceiling((float)NumH / threadCopyNum), 1);

                }
            }

            // find the split points for Vertical zones
            {
                // set the output region
                rvRegionsInput.SetViewsToResources(uavHRegions);
                rvRegionsOutput.SetViewsToResources(uavVRegions);
                rvInput.SetViewsToResources(uavBuffer3);

                // update the CB
                CB_Split localCB = new CB_Split();
                localCB.NumElements = (uint)MaxNumInputElements;
                localCB.NumRegionsH = (uint)HorizontalRegions;
                localCB.NumRegionsV = (uint)VerticalRegions;
                localCB.SplitOffset = (uint)VerticalRegionsOffset;
                CBFindSplitIndexV.UpdateValue(localCB);

                // execute the splitting
                CSFindSplitIndexV.Execute((int)Math.Ceiling((float)VerticalRegions / threadNum),
                                          (int)Math.Ceiling((float)HorizontalRegions / threadNum_V_Y));

                rvRegionsInput.SetViewsToResources(uavVRegions);
                rvRegionsOutput.SetViewsToResources(uavHRegions);

                localCB.NumRegionsH = (uint)RegionsNum; // we insert the over num regions in H
                CBFillRegionInfo.UpdateValue(localCB);

                // fill the region info
                CSFillRegionInfo.Execute((int)Math.Ceiling((float)RegionsNum / threadNum), 1);

            }

        } 

        #endregion


    }
}
