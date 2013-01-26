using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

// SharpDX lib
using SharpDX;

// internals libraries
using GraphicsEngine.Core.PrimaryObjects3D;
using GraphicsEngine.Import.FBX_Import;
using GraphicsEngine.Core;
using GraphicsEngine.Core.Shaders;
using GraphicsEngine.Managers;
using System.IO;

namespace GraphicsEngine.Import {
    public static class MeshImport {

        /// <summary>
        /// Load FBX Files from max 
        /// It loads all the meshes in the FBX files 
        /// including their textures
        /// </summary>
        /// <param name="FileName">The file name of the FBX file to load</param>
        public static void LoadFBX(String FileName)
        {
            String localFile = Environment.CurrentDirectory;
            Environment.CurrentDirectory = Path.GetDirectoryName(FileName);

            StringBuilder strB = new StringBuilder(FileName);

            /// start FBX dll file
            FBX.Initialize(strB);

            /// get the number of mesh that FBX have
            int meshNum = FBX.GetNumberMesh();

            /// load all the mesh
            for (int mesh_ind = 0; mesh_ind < meshNum; mesh_ind++) {
                /// make a new mesh
                Mesh mesh = new Mesh();

                /// add the mesh to the engine mesh list
                Engine.g_MeshManager.AddMesh(mesh);

                Vector3_t v3;

                /// select the mesh to the FBX 
                /// after that we can get anything 
                /// that we want from the mesh
                FBX.SelectMesh(mesh_ind);

                /// get the number of polygons
                int PoligonsNum = FBX.GetPolygons_Count();

                /// get the polygons and add them to the mesh
                for (int i = 0; i < PoligonsNum; i++) {
                    Polygon poly = FBX.GetPolygon(i);
                    mesh.AddPolygon(poly, false);
                }

                /// get the position of the mesh
                v3 = FBX.GetPosition();
                Vector3 pos = new Vector3(v3.x, v3.y, -v3.z);
                mesh.SetPosition(pos);

                /// get the Scale of the mesh
                v3 = FBX.GetScale();
                Vector3 scale = new Vector3(v3.x, v3.y, v3.z);
                mesh.SetScale(scale);

                /// get the rotation of the mesh
                v3 = FBX.GetRotation();
                Vector3 rot = new Vector3(-v3.x, -v3.z, -v3.y);
                rot *= (float)Math.PI;
                rot /= 180.0f;
                mesh.SetRotation(rot);

                mesh.SetRotation(new Vector3 (-(float)Math.PI/2f,0,0));

                /// get the material that the mesh have
                int MaterialCount = FBX.GetMaterialCount();

                /// Material name will be the same with the shader
                StringBuilder MaterialName = new StringBuilder(124);

                /// get the material name
                FBX.GetMaterialName(0, MaterialName);

                if (!ShaderManager.IsExist(MaterialName.ToString())) {

                    // create the shader
                    ShaderSimple sh = new ShaderSimple();

                    /// set to the new mesh the shader 
                    mesh.m_shader = sh;

                    /// add the shader to the list
                    ShaderManager.AddShader(MaterialName.ToString(), sh);

                    if (MaterialCount > 0) {
                        /// Get the number of the texture that the mesh has
                        /// Run through all textures found and add them
                        /// to the mesh according to their attributes
                        for (int i = 0; i < FBX.GetTextureCount(0); i++) {
                            StringBuilder TextureBld = new StringBuilder(124);
                            StringBuilder PropertyBld = new StringBuilder(124);

                            FBX.GetTexture(0, i, PropertyBld, TextureBld);
                            Console.WriteLine(
                                "Texture:"
                                + TextureBld.ToString()
                                + "  Property:"
                                + PropertyBld.ToString());

                            String PropertyStr = PropertyBld.ToString();
                            if (!TextureBld.ToString().Replace(" ", "").Equals("")) {
                                /// pass the textures
                                if (PropertyStr.Equals("DiffuseColor"))
                                    sh.SetTexture(TextureBld.ToString(), TextureType.Diffuse);
                                else if (PropertyStr.Equals("Bump"))
                                    sh.SetTexture(TextureBld.ToString(), TextureType.Bump);
                                else if (PropertyStr.Equals("ReflectionColor"))
                                    sh.SetTexture(TextureBld.ToString(), TextureType.Normal);
                            } else {
                                sh.SetVariables(new Vector3(0, 0, 1), ShaderViariables.Diffuse);
                            }
                        }
                    }
                } else {
                    mesh.m_shader = ShaderManager.GetExistShader(MaterialName.ToString());
                }

                /// create the mesh and download it to the card
                mesh.CreateMesh();

                /// get the mesh Name from the FBX
                StringBuilder MeshName = new StringBuilder(124);
                FBX.GetMeshName(MeshName);

                /// set the mesh name 
                mesh.Name = MeshName.ToString();
            }

            Environment.CurrentDirectory = localFile;
        }

    }
}
