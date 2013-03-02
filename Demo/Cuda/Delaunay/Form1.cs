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
using FxMaths.Geometry;
using FxMaths.GUI;
using FxMaths.GMaps;
using FxMaths.Cuda;

namespace Delaunay
{
    public partial class Form1 : Form
    {
        FxCuda cuda;

        // Variables
        csThreadInfo[] threadInfo;
        RegionInfo[] regionInfo;
        cbThreadParam threadParam;

        CudaDeviceVariable<csThreadInfo>    d_threadInfo;
        CudaDeviceVariable<RegionInfo>      d_regionInfo;
        CudaDeviceVariable<cbThreadParam>   d_threadParam;
        CudaDeviceVariable<FxVector2f>      d_vertex;
        CudaDeviceVariable<csHalfEdge>      d_HalfEdgeList;
        CudaDeviceVariable<csBoundaryNode>  d_BoundaryList;
        CudaDeviceVariable<csFace>          d_FaceList;

        BitonicSort<FxVector2f> GPUSort;
        CudaKernel triangulation;
        CudaKernel regionSplitH;
        CudaKernel regionSplitV_Phase1;
        CudaKernel regionSplitV_Phase2;

        /// <summary>
        /// List with all vertex
        /// </summary>
        List<FxVector2f> listAllVertex;
        FxVector2f[] sorted_Vertex;

        /// <summary>
        /// The number of vertex that we try to triangulate.
        /// </summary>
        int NumVertex = 0;


        const int stackMaxSize = 64;

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
        int maxVertexPerRegion = 100;

        // number of multiprocessors that the device have
        int MultiProcessorCount = 0;

        public Form1()
        {
            InitializeComponent();
            cuda = new FxCuda(true);
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
                listAllVertex.Add(new FxVector2f(x,y));
                //listAllVertex.Add(new FxVector2f(num-i));
            }

            WriteLine("Add " + num.ToString() + "to vertex list");
        }

        private void button1_Click(object sender, EventArgs e)
        {
            triangulation = cuda.LoadPTX("Triangulation", "PTX", "Triangulation");
            regionSplitH = cuda.LoadPTX("RegionSplit", "PTX", "splitRegionH");
            regionSplitV_Phase1 = cuda.LoadPTX("RegionSplit", "PTX", "splitRegionV_phase1");
            regionSplitV_Phase2 = cuda.LoadPTX("RegionSplit", "PTX", "splitRegionV_phase2");

            // add a random points  TODO: add external source (ex. file)
            CreateRandomPoints(1024 * 32, new FxVector2f(0, 0), new FxVector2f(2000, 2000));


            #region Set the max face/he/ve/boundary

            NumVertex = listAllVertex.Count;

            // select the spliting numbers
            // find the split points 
            NumRegions = (int)Math.Ceiling((float)NumVertex / (float)maxVertexPerRegion);

            HorizontalRegions = (int)Math.Floor(Math.Sqrt(NumRegions));
            VerticalRegions = (int)Math.Floor((float)NumRegions / (float)HorizontalRegions);
            NumRegions = HorizontalRegions * VerticalRegions;

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
            d_threadInfo = threadInfo;
            d_regionInfo = regionInfo;
            d_threadParam = threadParam;


            d_FaceList = new CudaDeviceVariable<csFace>(maxFacesPerThread * NumRegions);
            d_BoundaryList = new CudaDeviceVariable<csBoundaryNode>(maxBoundaryNodesPerThread * NumRegions);
            d_HalfEdgeList = new CudaDeviceVariable<csHalfEdge>(maxHalfEdgePerThread * NumRegions);

            // Update the region info by sort the vertex


            // try to sort the list 
            GPUSort = new BitonicSort<FxVector2f>(cuda);

        }



        void WriteLine(String str)
        {
            // write to the form
            //Console_Text.Text += str + "\n";

            // write to the console
            Console.WriteLine(str);

            //Tester.TesterForm.UIConsole.WriteLine(str);
        }




        #region Sorting of the regions

        RegionInfo[] regionInfoH;
        RegionInfoDebug[] regionDebugH;

        private void SortPartitions()
        {
            /* copy the data to the HW */
            d_vertex = listAllVertex.ToArray();

            // prepare the sorting
            GPUSort.Prepare(NumVertex, new FxVector2f(float.MaxValue));

            // set the internal data
            GPUSort.SetData(d_vertex, 0, NumVertex);

            // sort the x axis
            GPUSort.Sort(true, 0);

            // copy the results back
            GPUSort.GetResults(d_vertex, 0, NumVertex);

            // find the split points 
            int HorizontalRegionsOffset = (int)Math.Floor((float)NumVertex / (float)HorizontalRegions);
            int VerticalRegionsOffset = (int)Math.Floor((float)HorizontalRegionsOffset / (float)VerticalRegions);

            // calculate the region informations for Horizontal regions
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


#if true
            regionDebugH = new RegionInfoDebug[HorizontalRegions];
            sorted_Vertex = d_vertex;
            for (int i = 0; i < HorizontalRegions; i++)
            {
                regionDebugH[i].start.X = sorted_Vertex[regionInfoH[i].VertexOffset].X;
                if (i + 1 < HorizontalRegions)
                    regionDebugH[i].end.X = sorted_Vertex[regionInfoH[i + 1].VertexOffset].X;
                else
                    regionDebugH[i].end.X = sorted_Vertex[sorted_Vertex.Length - 1].X;
            }
#endif

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
                GPUSort.SetData(d_vertex, regionInfoH[i].VertexOffset, (int)regionInfoH[i].VertexNum);

                // sort the y axis
                GPUSort.Sort(true, 1);

                // copy the results back
                GPUSort.GetResults(d_vertex, regionInfoH[i].VertexOffset, (int)regionInfoH[i].VertexNum);

            }

            // create the region info list
            {
                regionSplitV_Phase1.BlockDimensions = new dim3(8, 8);
                regionSplitV_Phase1.GridDimensions = new dim3((int)Math.Ceiling((float)VerticalRegions / 8),
                                                              (int)Math.Ceiling((float)HorizontalRegions / 8));
                regionSplitV_Phase1.Run(d_vertex.DevicePointer,
                                        d_regionInfoH.DevicePointer,
                                        d_regionInfo.DevicePointer,
                                        HorizontalRegions,
                                        VerticalRegions,
                                        VerticalRegionsOffset);


                regionSplitV_Phase2.BlockDimensions = 32;
                regionSplitV_Phase2.GridDimensions = (int)Math.Ceiling((float)NumRegions / 32);
                regionSplitV_Phase2.Run(d_regionInfoH.DevicePointer,
                                        d_regionInfo.DevicePointer,
                                        HorizontalRegions,
                                        VerticalRegions,
                                        NumRegions,
                                        NumVertex);

            }

            // clean local temp gpu memorys
            d_regionInfoH.Dispose();
            GPUSort.Dispose();
        }
        
        #endregion








        private void RegionTriangulation()
        {
            /*
             *  const DATA_TYPE   *VertexList,
                HalfEdge          *HEList,
                BoundaryNode      *BoundaryList,
                Face              *FaceList,
                ThreadInfo        *threadInfoArray,
                const RegionInfo  *regionInfoArray,
                const ThreadParam  param,
                const int          RegionsNum)
             */ 

            // Start triangulation step 1
            triangulation.BlockDimensions = 128;
            triangulation.GridDimensions = (int)Math.Ceiling((float)NumRegions / 128);
            triangulation.Run(d_vertex.DevicePointer,
                              d_HalfEdgeList.DevicePointer,
                              d_BoundaryList.DevicePointer,
                              d_FaceList.DevicePointer,
                              d_threadInfo.DevicePointer, 
                              d_regionInfo.DevicePointer,
                              threadParam, 
                              NumRegions);

        }


        private void button2_Click(object sender, EventArgs e)
        {
            TimeStatistics.StartClock();
            SortPartitions();
            TimeStatistics.StopClock();

            TimeStatistics.StartClock();
            RegionTriangulation();
            TimeStatistics.StopClock();



            // debug info
            sorted_Vertex = d_vertex;
            for (int i = 0; i < 10; i++)
                Console.WriteLine(i.ToString() + " - " + sorted_Vertex[i].ToString() + " - " + listAllVertex[i].ToString());


            for (int i = 0; i < sorted_Vertex.Length; i++)
            {
                if (sorted_Vertex[i].x.Equals(float.NaN) || sorted_Vertex[i].y.Equals(float.NaN))
                {
                    Console.WriteLine("NaN on " + i.ToString());
                    break;
                }
            }

            
            DrawResults();


            // copy the data from the hardware
            threadInfo = d_threadInfo;

            d_threadInfo.Dispose();
            d_regionInfo.Dispose();
            d_threadParam.Dispose();

            GPUSort.Dispose();
            cuda.Dispose();
        }






        #region Debug Drawing

        private void DrawResults()
        {
            // draw points
            GeometryPlotElement plotElement = new GeometryPlotElement();
            foreach (FxVector2f v in sorted_Vertex)
            {
                Circle c = new Circle(v, 2);
                plotElement.AddGeometry(c, false);
            }


            // draw regions
            for (int i = 0; i < regionDebugH.Length; i++)
            {
                RegionInfoDebug r = regionDebugH[i];

                Line l = new Line(new FxVector2f(r.start.x, 0), new FxVector2f(r.start.x, 2000));
                l.LineColor = SharpDX.Color.AntiqueWhite;
                l.UseDefaultColor = false;
                plotElement.AddGeometry(l, false);
                l = new Line(new FxVector2f(r.end.x, 2000), new FxVector2f(r.end.x, 0));
                l.LineColor = SharpDX.Color.AntiqueWhite;
                l.UseDefaultColor = false;
                plotElement.AddGeometry(l, false);

            }

            // copy back the results of d_regions
            regionInfo = d_regionInfo;


            for (int i = 0; i < regionInfo.Length; i++)
            {
                RegionInfoDebug r = regionDebugH[(int)Math.Floor((float)i / VerticalRegions)];
                RegionInfo ri = regionInfo[i];

                FxVector2f v_start = sorted_Vertex[ri.VertexOffset];
                FxVector2f v_end = sorted_Vertex[ri.VertexOffset + ri.VertexNum];

                Line l = new Line(new FxVector2f(r.start.x, v_start.Y), new FxVector2f(r.end.x, v_start.Y));
                l.LineColor = SharpDX.Color.GreenYellow;
                l.UseDefaultColor = false;
                plotElement.AddGeometry(l, false);


                l = new Line(new FxVector2f(r.start.x, v_end.Y), new FxVector2f(r.end.x, v_end.Y));
                l.LineColor = SharpDX.Color.HotPink;
                l.UseDefaultColor = false;
                plotElement.AddGeometry(l, false);

            }

            /////// Draw triangles
            threadInfo = d_threadInfo;
            csFace[] faceList = d_FaceList;
            csHalfEdge[] heList = d_HalfEdgeList;

            for (int t = 0; t < threadInfo.Length; t++)
            {
                csThreadInfo tInfo = threadInfo[t];

                for (int f = 0; f < tInfo.lastFaceID.x; f++)
                {
                    csFace face = faceList[f + tInfo.lastFaceID.y];

                    csHalfEdge he1 = heList[face.halfEdgeID];
                    csHalfEdge he2 = heList[he1.nextEdgeID];
                    csHalfEdge he3 = heList[he2.nextEdgeID];

                    FxVector2f v1 = sorted_Vertex[he1.startVertexID];
                    FxVector2f v2 = sorted_Vertex[he2.startVertexID];
                    FxVector2f v3 = sorted_Vertex[he3.startVertexID];

                    Line l = new Line(v1,v2);
                    l.LineColor = SharpDX.Color.BlanchedAlmond;
                    l.UseDefaultColor = false;
                    l.LineWidth = 0.5f;
                    plotElement.AddGeometry(l, false);

                    l = new Line(v2, v3);
                    l.LineColor = SharpDX.Color.BlanchedAlmond;
                    l.UseDefaultColor = false;
                    l.LineWidth = 0.5f;
                    plotElement.AddGeometry(l, false);

                    l = new Line(v3, v1);
                    l.LineColor = SharpDX.Color.BlanchedAlmond;
                    l.UseDefaultColor = false;
                    l.LineWidth = 0.5f;
                    plotElement.AddGeometry(l, false);
                }
            }

            canvas1.AddElements(plotElement, false);
            canvas1.ReDraw();
        }
        
        #endregion

    }
}
