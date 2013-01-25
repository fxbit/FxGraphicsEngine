using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SlimDX;
using GraphicsEngine.Core;
using SlimDX.Direct3D11;
using System.Threading.Tasks;

namespace GraphicsEngine.Managers {
    public class Object3DManager {

        #region Variables

        /// <summary>
        /// List containing all meshes to be rendered
        /// </summary>
        public List<Object3D> g_Object3DList;

        /// <summary>
        /// Get the objects with index
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public Object3D this[int index]
        {
            get { return g_Object3DList[index]; }
        }

        #endregion

        #region Constructor
        public Object3DManager()
        {
            g_Object3DList = new List<Object3D>();
        }
        #endregion

        #region Functions

        /// <summary>
        /// Dispose all mesh resources
        /// </summary>
        public void Dispose()
        {
            /// run through the mesh list
            /// and dispose each mesh
            foreach (Object3D mesh in g_Object3DList)
                mesh.Dispose();
        }

        /// <summary>
        /// Render all objects
        /// </summary>
        public void Render( DeviceContext[] devCont )
        {

            if ( devCont.Length > 1 ) {
                int mod = devCont.Length;
                Task []renderTask= new Task[mod];

                for ( int i=0; i < mod; i++ ) {
                    renderTask[i] = new Task( ( m ) => {
                        int blockID = (int)m;

                        int size = (int)Math.Ceiling( g_Object3DList.Count / (float)mod );

                        for ( int j =size * blockID; j < size * ( blockID + 1 ); j++ ) {

                            if ( j < g_Object3DList.Count )
                                g_Object3DList[j].Render( devCont[blockID] );

                        }
                    }, i );


                    renderTask[i].Start();
                }

                // wait the threads
                for ( int i=0; i < mod; i++ ) {
                    renderTask[i].Wait();
                }
            } else {
                /// run through all meshes in mesh list 
                /// and Render them
                foreach ( Object3D mesh in g_Object3DList )
                    mesh.Render( devCont[0] );
            }
        }


        /// <summary>
        /// Add the object to the meshlist
        /// </summary>
        /// <param name="mesh"></param>
        public void AddMesh(Object3D mesh)
        {
            g_Object3DList.Add(mesh);
        }

        #endregion

    }
}
