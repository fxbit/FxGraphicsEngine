using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace GraphicsEngine.Core.PrimaryObjects3D
{
    public class Sphere : Mesh
    {

        /// <summary>
        /// Create a new sphere with specific radius 
        /// </summary>
        /// <param name="radius"></param>
        /// <param name="zones">how many slites in z axes</param>
        /// <param name="sections"></param>
        public Sphere( float radius, int zones, int sections ) : base()
        {
            // set zero position
            base.SetPosition( new SlimDX.Vector3() );

            // create the sphere 


        }
    }
}
