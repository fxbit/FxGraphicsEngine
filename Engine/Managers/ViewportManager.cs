using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using GraphicsEngine.Core;

namespace GraphicsEngine.Managers {
    public static class ViewportManager {

        /// <summary>
        /// The list of all viewport
        /// </summary>
        public static List<Viewport> ListOfViewport = new List<Viewport>();


        public static void AddViewport(Viewport viewport)
        {
            ListOfViewport.Add(viewport);
        }

        public static void RemoveViewport(Viewport viewport)
        {
            ListOfViewport.Remove(viewport);
        }
    }
}
