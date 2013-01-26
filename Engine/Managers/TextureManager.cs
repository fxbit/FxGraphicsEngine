using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using GraphicsEngine.Core;
using SharpDX.Direct3D11;

namespace GraphicsEngine.Managers {
    public static class TextureManager {

        /// <summary>
        /// The list of all shader 
        /// in dictionary way
        /// </summary>
        private static Dictionary<String, Texture> ListOfTextures = new Dictionary<String, Texture>();

        public static Texture GetExistTexture(String key)
        {
            /// return the shader 
            return ListOfTextures[key];
        }

        public static void AddTexture(String Key, Texture texture)
        {
            /// check if the key is all ready exist
            if (!ListOfTextures.ContainsKey(Key)) {
                /// Add the shader to the list with the name for key
                ListOfTextures.Add(Key, texture);
            }
        }

        public static Texture AddTexture(String Path)
        {
            /// check if the texture is all ready exist
            if (!ListOfTextures.ContainsKey(Path)) {

                /// create the texture base on the path
                Texture newTex = new Texture();
                newTex.Path = Path;
                newTex.ScaleU = 1f;
                newTex.ScaleV = 1f;
                newTex.Alpha = 1f;
                newTex.texture2D = (Texture2D)Texture2D.FromFile( Engine.g_device, Path);
                newTex.shaderResource = new ShaderResourceView(Engine.g_device, newTex.texture2D);

                /// add the new texture to the list
                ListOfTextures.Add(Path, newTex);

                // return the new texture
                return newTex;
            } else {
                /// return the texture
                return ListOfTextures[Path];
            }
        }

        public static void RemoveShader(String Key)
        {
            /// remove the shader with the specific key
            ListOfTextures.Remove(Key);
        }
    }
}
