using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

using ManagedCuda;
using ManagedCuda.NPP;
using ManagedCuda.VectorTypes;

using FxMaths;
using FxMaths.Vector;
using FxMaths.GMaps;

namespace Delaunay
{
    public partial class Form1 : Form
    {
        FxCuda cuda;

        // Variables
        csThreadInfo[] threadInfo;
        RegionInfo[] regionInfo;
        cbThreadParam threadParam;

        CudaDeviceVariable<csThreadInfo>  d_threadInfo;
        CudaDeviceVariable<RegionInfo>    d_regionInfo;
        CudaDeviceVariable<cbThreadParam> d_threadParam;
        CudaDeviceVariable<FxVector2f>    d_vertex;

        MergeSort<FxVector2f> mergeSort;
        CudaKernel triangulation;
        CudaKernel regionSplitH;

        /// <summary>
        /// List with all vertex
        /// </summary>
        List<FxVector2f> listAllVertex;
        FxVector2f[] Vertex;

        /// <summary>
        /// The number of vertex that we try to triangulate.
        /// </summary>
        int NumVertex = 0;


        const int stackMaxSize = 64;

        const float MergeVXThread = 4.0f;
        const float MergeVYThread = 8.0f;
        const float MergeThread = 1.0f;
        const int TriangulationThread = 128;



        /// <summary>
        /// The num of subregions
        /// </summary>
        int NumRegions = 0;

        /// <summary>
        /// The number of horizontal regions
        /// </summary>
        int HorizontalRegions;

        /// <summary>
        /// The number of vertical regions per horizontal
        /// </summary>
        int VerticalRegions;


        // max faces per thread
        int maxFacesPerThread;

        // max Half edge per thread
        int maxHalfEdgePerThread;

        // max vertex per thread
        int maxBoundaryNodesPerThread;

        // the max number of vertex per region
        int maxVertexPerRegion=200;

        // number of multiprocessors that the device have
        int MultiProcessorCount = 0;

        public Form1()
        {
            InitializeComponent();
            cuda = new FxCuda();
            MultiProcessorCount = cuda.ctx.GetDeviceInfo().MultiProcessorCount;

            listAllVertex = new List<FxVector2f>();
        }


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

        private void button1_Click(object sender, EventArgs e)
        {
            triangulation = cuda.LoadPTX("Triangulation", "PTX", "Triangulation");
            regionSplitH = cuda.LoadPTX("RegionSplit", "PTX", "splitRegionH");

            // add a random points  TODO: add external source (ex. file)
            CreateRandomPoints(1024*64, new FxVector2f(0, 0), new FxVector2f(100000, 100000));


            #region Set the max face/he/ve/boundary

            NumVertex = listAllVertex.Count;

            // select the spliting numbers
            // find the split points 
            NumRegions          = (int)Math.Ceiling((float)NumVertex / (float)maxVertexPerRegion);

            HorizontalRegions   = (int)Math.Floor(Math.Sqrt(NumRegions));
            VerticalRegions     = (int)Math.Floor((float)NumRegions / (float)HorizontalRegions);
            NumRegions          = HorizontalRegions * VerticalRegions;

            // init the array sizes

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

            #endregion


            // init the array on cpu side
            threadInfo = new csThreadInfo[NumRegions];
            regionInfo = new RegionInfo[NumRegions];
            threadParam = new cbThreadParam();

            #region init the thread param

            // init the thread param
            threadParam.maxFacesPerThread = (uint)maxFacesPerThread;
            threadParam.maxHalfEdgePerThread = (uint)maxHalfEdgePerThread;
            threadParam.maxBoundaryNodesPerThread = (uint)maxBoundaryNodesPerThread;
            threadParam.RegionsNum = (uint)NumRegions;
            
            #endregion

            // copy the data to the hardware
            d_threadInfo    = threadInfo;
            d_regionInfo    = regionInfo;
            d_threadParam   = threadParam;

            // Update the region info by sort the vertex

            
            // try to sort the list 
            mergeSort = new MergeSort<FxVector2f>(cuda);

        }


        void WriteLine(String str)
        {
            // write to the form
            //Console_Text.Text += str + "\n";

            // write to the console
            Console.WriteLine(str);

            //Tester.TesterForm.UIConsole.WriteLine(str);
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        RegionInfo[] regionInfoH;

        private void SortPartitions()
        {
            /* copy the data to the HW */
            d_vertex = listAllVertex.ToArray();

            // prepare the sorting
            mergeSort.Prepare(NumVertex);

            // set the internal data
            mergeSort.SetData(d_vertex, 0, NumVertex, d_vertex.TypeSize);

            // sort the x axis
            mergeSort.Sort(true, 0);

            // copy the results back
            mergeSort.GetResults(d_vertex, 0, NumVertex, d_vertex.TypeSize);

            // find the split points 
            int HorizontalRegionsOffset = (int)Math.Floor((float)NumVertex / (float)HorizontalRegions);
            int VerticalRegionsOffset = (int)Math.Floor((float)HorizontalRegionsOffset / (float)VerticalRegions);


            // calculate the region informations for Horizontal regions
            {
                regionInfoH = new RegionInfo[HorizontalRegions];
                CudaDeviceVariable<RegionInfo> d_regionInfoH = regionInfoH;

                int blockDim = (int)Math.Ceiling((double)HorizontalRegions / MultiProcessorCount);
                regionSplitH.BlockDimensions = blockDim;
                regionSplitH.GridDimensions = (int)Math.Ceiling((double)HorizontalRegions / blockDim);
                regionSplitH.Run(d_vertex.DevicePointer,
                                 d_regionInfoH.DevicePointer,
                                 HorizontalRegions,
                                 HorizontalRegionsOffset,
                                 NumVertex);

                regionInfoH = d_regionInfoH;
                d_regionInfoH.Dispose();
            }

            // sort each subregion based on y-axes
            for (int i = 0; i < HorizontalRegions; i++)
            {
                // update vertex number
                if (i < HorizontalRegions - 1)
                {
                    regionInfoH[i].VertexNum = regionInfoH[i + 1].VertexOffset - regionInfoH[i].VertexOffset;
                }
                else
                {
                    regionInfoH[i].VertexNum = (uint)(NumVertex - regionInfoH[i].VertexOffset);
                }

                // set the internal data
                mergeSort.SetData(d_vertex, regionInfoH[i].VertexOffset, (int)regionInfoH[i].VertexNum, d_vertex.TypeSize);

                // sort the y axis
                mergeSort.Sort(true, 1);

                // copy the results back
                mergeSort.GetResults(d_vertex, regionInfoH[i].VertexOffset, (int)regionInfoH[i].VertexNum, d_vertex.TypeSize);
            }

            // create the region info list
            {



            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            TimeStatistics.StartClock();
            SortPartitions();
            TimeStatistics.StopClock();

            // debug info
            FxVector2f[] results = d_vertex;
            for (int i = 0; i < 10; i++)
                Console.WriteLine(results[i].ToString() + " - " + listAllVertex[i].ToString());
           
            // Invoke kernel
            triangulation.BlockDimensions = TriangulationThread;
            triangulation.GridDimensions = (NumRegions + TriangulationThread - 1) / TriangulationThread;
            triangulation.Run(d_threadInfo.DevicePointer, d_regionInfo.DevicePointer, d_threadParam.DevicePointer, NumRegions);

            // copy the data from the hardware
            threadInfo = d_threadInfo;

            d_threadInfo.Dispose();
            d_regionInfo.Dispose();
            d_threadParam.Dispose();

            mergeSort.Dispose();
            cuda.Dispose();
        }


    }
}
