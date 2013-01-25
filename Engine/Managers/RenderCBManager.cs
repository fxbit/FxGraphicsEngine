using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace GraphicsEngine.Managers
{
    /// <summary>
    /// The call back delegate function
    /// </summary>
    public delegate void RenderCBFunction();
    
    /// <summary>
    /// Select the timing for the processing
    /// </summary>
    public enum RenderCBTiming { PreProcessing, PostProcessing };

    public static class RenderCBManager
    {
        /// <summary>
        /// The list of all cb for pre processing
        /// </summary>
        public static List<RenderCBFunction> ListOfPrePCB = new List<RenderCBFunction>();

        /// <summary>
        /// The list of all cb for post processing
        /// </summary>
        public static List<RenderCBFunction> ListOfPostPCB = new List<RenderCBFunction>();

        /// <summary>
        /// Add cb for specific timing
        /// </summary>
        /// <param name="cb"></param>
        /// <param name="timing"></param>
        public static void AddCB(RenderCBFunction cb, RenderCBTiming timing)
        {
            // selecet the timing of the cb
            switch ( timing ) {
                case  RenderCBTiming.PostProcessing:
                    // add the cb to the post list
                    ListOfPostPCB.Add( cb );
                    break;
                case RenderCBTiming.PreProcessing:
                    // add the cb to the pre list
                    ListOfPrePCB.Add( cb );
                    break;

            }
        }

        /// <summary>
        /// Run the cb 
        /// </summary>
        /// <param name="timing"></param>
        public static void RunCB( RenderCBTiming timing )
        {
            switch ( timing ) {
                case RenderCBTiming.PostProcessing:
                    // call all the cb's
                    foreach ( RenderCBFunction cb in ListOfPostPCB )
                        cb();
                    break;
                case RenderCBTiming.PreProcessing:
                    // call all the cb's
                    foreach ( RenderCBFunction cb in ListOfPrePCB )
                        cb();
                    break;

            }
        }
    }
}
