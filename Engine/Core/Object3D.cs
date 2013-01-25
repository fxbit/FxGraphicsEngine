
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SlimDX;
using SlimDX.Direct3D11;

namespace GraphicsEngine.Core {
    public abstract class Object3D {
        /// <summary>
        /// Boundary box
        /// </summary>
        public BoundingBox m_BoundaryBox;

        /// <summary>
        /// matrix for the position of object
        /// </summary>
        public Matrix m_WorldMatrix;

        /// <summary>
        /// The name of the object...
        /// for interface reason's
        /// </summary>
        public String Name = "";

        /// <summary>
        /// Draw the object on screen
        /// </summary>
        public abstract void Render(DeviceContext devCont);

        /// <summary>
        /// Release all resources of the object
        /// </summary>
        public abstract void Dispose();

        /// <summary>
        /// Refresh parameters of object, used for animating
        /// </summary>
        public abstract void Update();
    }
}
