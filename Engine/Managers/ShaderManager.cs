using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using GraphicsEngine.Core;

namespace GraphicsEngine.Managers {
    public static class ShaderManager {

        /// <summary>
        /// The list of all shader 
        /// in dictionary way
        /// </summary>
        private static Dictionary<String, Shader> ListOfShader = new Dictionary<string, Shader>();

        public static Boolean IsExist(String Key)
        {
            return ListOfShader.ContainsKey(Key);
        }

        public static Shader GetExistShader(String key)
        {
            /// return the shader 
            return ListOfShader[key];
        }

        public static void AddShader(String Key, Shader shader)
        {
            /// check if the key is all ready exist
            if (!ListOfShader.ContainsKey(Key)) {
                /// Add the shader to the list with the name for key
                ListOfShader.Add(Key, shader);
            }
        }

        public static void RemoveShader(String Key)
        {
            /// remove the shader with the specific key
            ListOfShader.Remove(Key);
        }
    }
}
