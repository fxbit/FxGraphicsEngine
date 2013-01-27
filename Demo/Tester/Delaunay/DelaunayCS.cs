
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

using GraphicsEngine.Core;

using FxMaths;
using FxMaths.GMaps;
using FxMaths.Vector;

using FXFramework;

using SharpDX.Direct3D11;

using Buffer        = SharpDX.Direct3D11.Buffer;
using D3D           = SharpDX.Direct3D11;
using ComputeShader = GraphicsEngine.Core.ComputeShader;
using GraphicsEngine;

namespace Delaunay
{
    class DelaunayCS
    {
        ////////////////////////////////////// Parameters

        int NumVertex = 1000000;
        const int stackMaxSize = 64;
        int MaxPointsPerRegion = 500;

        const float MergeVXThread = 4.0f;
        const float MergeVYThread = 8.0f;
        const float MergeThread = 1.0f;
        const float TriangulationThread = 128.0f;

        ////////////////////////////////////// Variables


        #region Variables

        /// <summary>
        /// List with all vertex
        /// </summary>
        List<IVertex<float>> listAllVertex;

        /// <summary>
        /// Compute Shader for triangulation
        /// </summary>
        ComputeShader CSSubRegions = null;

        /// <summary>
        /// Compute Shader for the vertical merging
        /// </summary>
        ComputeShader CSVMerging = null;

        /// <summary>
        /// Compute Shader for the Horizontal merging
        /// </summary>
        ComputeShader CSHMerging = null;

        /// <summary>
        /// The num of subregions
        /// </summary>
        int NumRegions = 0;

        /// <summary>
        /// The Max Num Points Per Region
        /// </summary>
        int PointsPerRegion = 0;

        /// <summary>
        /// The number of horizontal regions
        /// </summary>
        int HorizontalRegions;

        /// <summary>
        /// The number of vertical regions per horizontal
        /// </summary>
        int VerticalRegions;


        /// <summary>
        /// The search algorithm
        /// </summary>
        BitonicSort bitonicSort;

        #endregion



        #region CS Variables

        // max faces per thread
        int maxFacesPerThread;

        // max Half edge per thread
        int maxHalfEdgePerThread;


        // max vertex per thread
        int maxBoundaryNodesPerThread;

        // num of dispatch kernels
        int kenrelNum;

        // num of dispatch kernels for the vertical merging
        int mergeVXkernelNum;
        int mergeVYkernelNum;

        // num of the merging threads for the vertical axes
        int mergeVthreadNum;
        int mergeVXthreadNum;
        int mergeVYthreadNum;

        // num of dispatch kernels for the Horizontal merging
        int mergeHkernelNum;

        // num of the merging threads for the Horizontal axes
        int mergeHthreadNum;

        // the max number of vertex per region
        int maxVertexPerRegion;

        Buffer InputPoints;
        Buffer RegionInfo;
        Buffer FaceListBuffer;
        Buffer HalfEdgeListBuffer;
        Buffer BoundaryListBuffer;
        Buffer ThreadInfoListBuffer;
        Buffer DelaunayStackBuffer;
        Buffer UintStackBuffer;


        FxResourceVariableList rvlInputPoints;
        FxResourceVariableList rvlRegionInfo;
        FxResourceVariableList rvlFaceListBuffer;
        FxResourceVariableList rvlHalfEdgeListBuffer;
        FxResourceVariableList rvlBoundaryListBuffer;
        FxResourceVariableList rvlThreadInfoListBuffer;
        FxResourceVariableList rvlDelaunayStackBuffer;
        FxResourceVariableList rvlUintStackBuffer;

        ShaderResourceView srvInputPoints;
        ShaderResourceView srvRegionInfo;
        UnorderedAccessView uavFaceListBuffer;
        UnorderedAccessView uavHalfEdgeListBuffer;
        UnorderedAccessView uavBoundaryListBuffer;
        UnorderedAccessView uavThreadInfoListBuffer;
        UnorderedAccessView uavDelaunayStackBuffer;
        UnorderedAccessView uavUintStackBuffer;

        Buffer stagingInputPointsBuffer;
        Buffer stagingFaceListBuffer;
        Buffer stagingHalfEdgeListBuffer;
        Buffer stagingBoundaryListBuffer;
        Buffer stagingThreadInfoListBuffer;
        Buffer stagingDelaunayListBuffer;
        Buffer stagingUintStackBuffer;

        #endregion



        public DelaunayCS()
        {
            // init vertex list 
            listAllVertex = new List<IVertex<float>>();

        }

        #region Vertex List Management

        /// <summary>
        /// Create given number of uniform random points 
        /// on a given range(min,max)
        /// </summary>
        /// <param name="num">The number of the random points</param>
        /// <param name="min">The min position of the range that the points are going to generate</param>
        /// <param name="max">The max position of the range that the points are going to generate</param>
        public void CreateRandomPoints(int num, FxVector2f min, FxVector2f max)
        {
            // generate and add the points 
            float x, y;
            Random rand = new Random();
            for (int i = 0; i < num; i++)
            {
                x = min.X + (float)(rand.NextDouble() * (max.X - min.X));
                y = min.Y + (float)(rand.NextDouble() * (max.Y - min.Y));

                //canvas1 add a new vertex
                listAllVertex.Add(new FxVector2f(x, y));
            }

            WriteLine("Add " + num.ToString() + "to vertex list");
        }

        /// <summary>
        /// Clean the internal list of vertex.
        /// </summary>
        public void ClearVertex()
        {
            // clean the prev list
            listAllVertex.Clear();
        } 

        #endregion


        #region Init Data


        private void InitDataForCS()
        {


            #region Set the max face/he/ve/boundary

            // max faces per thread
            maxFacesPerThread = maxVertexPerRegion * 3;
            maxFacesPerThread += maxFacesPerThread % 32;

            // max Half edge per thread
            maxHalfEdgePerThread = maxFacesPerThread * 4;
            maxHalfEdgePerThread += maxHalfEdgePerThread % 32;

            // max vertex per thread
            maxBoundaryNodesPerThread = maxVertexPerRegion * 3;
            maxBoundaryNodesPerThread += maxBoundaryNodesPerThread % 32;

            WriteLine("maxFacesPerThread:" + maxFacesPerThread.ToString());
            WriteLine("maxHalfEdgePerThread:" + maxHalfEdgePerThread.ToString());
            WriteLine("maxBoundaryNodesPerThread:" + maxBoundaryNodesPerThread.ToString());

            kenrelNum = (int)Math.Ceiling(NumRegions / TriangulationThread);
            WriteLine("kenrelNum:" + kenrelNum.ToString());


            #endregion



            #region pass the Vertex data

            // the input buffer will be allocate from bitonic sort

            rvlInputPoints.AddResourceFromShader(CSSubRegions.m_effect, "InputVertex");
            rvlInputPoints.AddResourceFromShader(CSVMerging.m_effect, "InputVertex");
            rvlInputPoints.AddResourceFromShader(CSHMerging.m_effect, "InputVertex");


            #endregion



            #region pass the Region data


            rvlRegionInfo.AddResourceFromShader(CSSubRegions.m_effect, "RegionInfoBuffer");

            #endregion



            #region Create the FaceList output buffers

            // create the input mesh with the faces that we want
            FaceListBuffer = ComputeShader.CreateBuffer(maxFacesPerThread * NumRegions, csFace.GetStructSize(), AccessViewType.UAV);
            FaceListBuffer.DebugName = "FaceListBuffer";

            // create the UAV
            uavFaceListBuffer = FXResourceVariable.InitUAVResource(Engine.g_device, FaceListBuffer);

            // add the buffer to the shader resource list
            rvlFaceListBuffer.AddResourceFromShader(CSSubRegions.m_effect, "FaceList");
            rvlFaceListBuffer.AddResourceFromShader(CSVMerging.m_effect, "FaceList");
            rvlFaceListBuffer.AddResourceFromShader(CSHMerging.m_effect, "FaceList");

            #endregion



            #region Create the HalfEdgeList output buffers

            // create the input mesh with the faces that we want
            HalfEdgeListBuffer = ComputeShader.CreateBuffer(maxHalfEdgePerThread * NumRegions, csHalfEdge.GetStructSize(), AccessViewType.UAV);
            HalfEdgeListBuffer.DebugName = "HalfEdgeListBuffer";


            uavHalfEdgeListBuffer = FXResourceVariable.InitUAVResource(Engine.g_device, HalfEdgeListBuffer);

            rvlHalfEdgeListBuffer.AddResourceFromShader(CSSubRegions.m_effect, "HalfEdgeList");
            rvlHalfEdgeListBuffer.AddResourceFromShader(CSVMerging.m_effect, "HalfEdgeList");
            rvlHalfEdgeListBuffer.AddResourceFromShader(CSHMerging.m_effect, "HalfEdgeList");

            #endregion



            #region Create the BoundaryList output buffers

            // create the input mesh with the faces that we want
            BoundaryListBuffer = ComputeShader.CreateBuffer(maxBoundaryNodesPerThread * NumRegions, csBoundaryNode.GetStructSize(), AccessViewType.UAV);
            BoundaryListBuffer.DebugName = "BoundaryListBuffer";


            uavBoundaryListBuffer = FXResourceVariable.InitUAVResource(Engine.g_device, BoundaryListBuffer);

            rvlBoundaryListBuffer.AddResourceFromShader(CSSubRegions.m_effect, "BoundaryList");
            rvlBoundaryListBuffer.AddResourceFromShader(CSVMerging.m_effect, "BoundaryList");
            rvlBoundaryListBuffer.AddResourceFromShader(CSHMerging.m_effect, "BoundaryList");

            #endregion



            #region Create the ThreadInfoList output buffers

            // create the input mesh with the faces that we want
            ThreadInfoListBuffer = ComputeShader.CreateBuffer(NumRegions, csThreadInfo.GetStructSize(), AccessViewType.UAV);
            ThreadInfoListBuffer.DebugName = "ThreadInfoList";

            uavThreadInfoListBuffer = FXResourceVariable.InitUAVResource(Engine.g_device, ThreadInfoListBuffer);

            rvlThreadInfoListBuffer.AddResourceFromShader(CSSubRegions.m_effect, "ThreadInfoList");
            rvlThreadInfoListBuffer.AddResourceFromShader(CSVMerging.m_effect, "ThreadInfoList");
            rvlThreadInfoListBuffer.AddResourceFromShader(CSHMerging.m_effect, "ThreadInfoList");

            #endregion



            #region Create the DelaunayStackBuffer output buffers

            // create the input mesh with the faces that we want
            DelaunayStackBuffer = ComputeShader.CreateBuffer(NumRegions * 50, csDelaunayNode.GetStructSize(), AccessViewType.UAV);
            DelaunayStackBuffer.DebugName = "DelaunayStackBuffer";

            uavDelaunayStackBuffer = FXResourceVariable.InitUAVResource(Engine.g_device, DelaunayStackBuffer);

            rvlDelaunayStackBuffer.AddResourceFromShader(CSSubRegions.m_effect, "DeleanayNodeStack");
            rvlDelaunayStackBuffer.AddResourceFromShader(CSVMerging.m_effect, "DeleanayNodeStack");
            rvlDelaunayStackBuffer.AddResourceFromShader(CSHMerging.m_effect, "DeleanayNodeStack");

            #endregion



            #region Create the UintStack output buffers

            UintStackBuffer = ComputeShader.CreateBuffer(2 * NumRegions * stackMaxSize, ComputeShader.SizeOfInt1, AccessViewType.UAV);
            UintStackBuffer.DebugName = "UintStackBuffer";

            uavUintStackBuffer = FXResourceVariable.InitUAVResource(Engine.g_device, UintStackBuffer);

            rvlUintStackBuffer.AddResourceFromShader(CSVMerging.m_effect, "UintStack");
            rvlUintStackBuffer.AddResourceFromShader(CSHMerging.m_effect, "UintStack");

            #endregion



            #region Staging Buffers

            stagingThreadInfoListBuffer = ComputeShader.CreateStagingBuffer(ThreadInfoListBuffer);
            stagingHalfEdgeListBuffer = ComputeShader.CreateStagingBuffer(HalfEdgeListBuffer);
            stagingBoundaryListBuffer = ComputeShader.CreateStagingBuffer(BoundaryListBuffer);
            stagingFaceListBuffer = ComputeShader.CreateStagingBuffer(FaceListBuffer);
            stagingDelaunayListBuffer = ComputeShader.CreateStagingBuffer(DelaunayStackBuffer);
            stagingUintStackBuffer = ComputeShader.CreateStagingBuffer(UintStackBuffer);

            #endregion
        }


        #endregion


        #region Init Shader


        public void InitShaders(Device device)
        {
            WriteLine("MaxPointsPerRegion : " + MaxPointsPerRegion.ToString());


            // select the spliting numbers
            // find the split points 
            PointsPerRegion = MaxPointsPerRegion;
            NumRegions = (int)Math.Ceiling((float)NumVertex / (float)PointsPerRegion);

            HorizontalRegions = (int)Math.Floor(Math.Sqrt(NumRegions));
            VerticalRegions = (int)Math.Floor((float)NumRegions / (float)HorizontalRegions);

            // recalc the region number
            NumRegions = HorizontalRegions * VerticalRegions;

            // inc the max vertex per region
            maxVertexPerRegion = (int)((float)MaxPointsPerRegion * 1.3f);

            // calc the threads
            mergeVYthreadNum = HorizontalRegions;
            mergeVXthreadNum = VerticalRegions - 1;
            mergeVthreadNum = mergeVXthreadNum * mergeVYthreadNum;
            mergeVXkernelNum = (int)Math.Ceiling(mergeVXthreadNum / MergeVXThread);
            mergeVYkernelNum = (int)Math.Ceiling(mergeVYthreadNum / MergeVYThread);

            mergeHthreadNum = HorizontalRegions - 1;
            mergeHkernelNum = (int)Math.Ceiling(mergeHthreadNum / MergeThread);

            TimeStatistics.StartClock();

            // compile the shader
            CSSubRegions = new ComputeShader(@"Delaunay\Shaders\Triangulation.hlsl", "main",@"Delaunay\Shaders\");
            CSVMerging = new ComputeShader(@"Delaunay\Shaders\MergeVertical.hlsl", "main", @"Delaunay\Shaders\");
            CSHMerging = new ComputeShader(@"Delaunay\Shaders\MergeHorizontal.hlsl", "main", @"Delaunay\Shaders\");
            bitonicSort = new BitonicSort(NumVertex, device);

            float time = TimeStatistics.ClockLap("Load Shaders");
            WriteLine("Load Shaders:" + time.ToString());

            TimeStatistics.StartClock();

            // init variables list
            rvlInputPoints = new FxResourceVariableList();
            rvlRegionInfo = new FxResourceVariableList();
            rvlFaceListBuffer = new FxResourceVariableList();
            rvlHalfEdgeListBuffer = new FxResourceVariableList();
            rvlBoundaryListBuffer = new FxResourceVariableList();
            rvlThreadInfoListBuffer = new FxResourceVariableList();
            rvlDelaunayStackBuffer = new FxResourceVariableList();
            rvlUintStackBuffer = new FxResourceVariableList();

            // init the data 
            InitDataForCS();

            // init the data
            bitonicSort.InitBuffers(out InputPoints, out RegionInfo);

            // set the data
            bitonicSort.FillData(listAllVertex);

            srvInputPoints = FXResourceVariable.InitSRVResource(device, InputPoints);
            srvRegionInfo = FXResourceVariable.InitSRVResource(device, RegionInfo); // this is complex :P
            stagingInputPointsBuffer = ComputeShader.CreateStagingBuffer(InputPoints);

            // link the resource with buffers
            rvlInputPoints.SetViewsToResources(srvInputPoints);
            rvlRegionInfo.SetViewsToResources(srvRegionInfo);
            rvlFaceListBuffer.SetViewsToResources(uavFaceListBuffer);
            rvlHalfEdgeListBuffer.SetViewsToResources(uavHalfEdgeListBuffer);
            rvlBoundaryListBuffer.SetViewsToResources(uavBoundaryListBuffer);
            rvlThreadInfoListBuffer.SetViewsToResources(uavThreadInfoListBuffer);
            rvlDelaunayStackBuffer.SetViewsToResources(uavDelaunayStackBuffer);
            rvlUintStackBuffer.SetViewsToResources(uavUintStackBuffer);

            time = TimeStatistics.ClockLap("Load Data");
            WriteLine("Load Data:" + time.ToString());

            WriteLine("===================================== Size in bytes:");
            WriteLine("InputPoints Size = " + InputPoints.Description.SizeInBytes.ToString());
            WriteLine("RegionInfo Size = " + RegionInfo.Description.SizeInBytes.ToString());
            WriteLine("FaceListBuffer Size = " + FaceListBuffer.Description.SizeInBytes.ToString());
            WriteLine("HalfEdgeListBuffer Size = " + HalfEdgeListBuffer.Description.SizeInBytes.ToString());
            WriteLine("BoundaryListBuffer Size = " + BoundaryListBuffer.Description.SizeInBytes.ToString());
            WriteLine("ThreadInfoListBuffer Size = " + ThreadInfoListBuffer.Description.SizeInBytes.ToString());
            WriteLine("DelaunayStackBuffer Size = " + DelaunayStackBuffer.Description.SizeInBytes.ToString());
            WriteLine("UintStackBuffer Size = " + UintStackBuffer.Description.SizeInBytes.ToString());
            float sum = InputPoints.Description.SizeInBytes
                        + RegionInfo.Description.SizeInBytes
                        + FaceListBuffer.Description.SizeInBytes
                        + HalfEdgeListBuffer.Description.SizeInBytes
                        + BoundaryListBuffer.Description.SizeInBytes
                        + ThreadInfoListBuffer.Description.SizeInBytes
                        + DelaunayStackBuffer.Description.SizeInBytes
                        + UintStackBuffer.Description.SizeInBytes;

            WriteLine("===================================== Overall size in bytes:" + sum.ToString());
            WriteLine("===================================== Overall size in Mbytes:" + (sum / (1024 * 1024)).ToString());


        } 

        #endregion


        void WriteLine(String str)
        {
            // write to the form
            //Console_Text.Text += str + "\n";

            // write to the console
            Console.WriteLine(str);

            Tester.TesterForm.UIConsole.WriteLine(str);
        }
    }
}
