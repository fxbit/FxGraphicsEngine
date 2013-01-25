using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using SlimDX.Direct3D11;

namespace GraphicsEngine.Core
{
    public class Profiling
    {
        private String Name;


        private Query DisjointQuery;
        private Query TimestampStartQuery;
        private Query TimestampEndQuery;
        bool QueryStarted;
        bool QueryFinished;

        public Profiling()
        {
            QueryStarted = false;
            QueryFinished = false;
        }


        public void StartProfile(String name)
        {
            // set the name of the profiling
            Name = name;

            QueryDescription desc = new QueryDescription();
            desc.Flags = QueryFlags.None;
            desc.Type = QueryType.TimestampDisjoint;
            DisjointQuery = new Query(Engine.g_device, desc);

            desc.Flags = QueryFlags.None;
            desc.Type = QueryType.Timestamp;
            TimestampStartQuery = new Query(Engine.g_device, desc);
            TimestampEndQuery = new Query(Engine.g_device, desc);

            // Start a disjoint query first
            Engine.g_device.ImmediateContext.Begin(DisjointQuery);

            // Insert the start timestamp  
            Engine.g_device.ImmediateContext.Begin(TimestampStartQuery);

            QueryStarted = true;
        }

        public void EndProfile()
        {
            // Insert the end timestamp    
            Engine.g_device.ImmediateContext.End(TimestampEndQuery);

            // End the disjoint query
            Engine.g_device.ImmediateContext.End(DisjointQuery);

            QueryStarted = false;
            QueryFinished = true;

            // collect data
            Engine.g_device.ImmediateContext.GetData(TimestampStartQuery);
            Engine.g_device.ImmediateContext.GetData(TimestampEndQuery);
            Engine.g_device.ImmediateContext.GetData(DisjointQuery);




        }
    }
}
