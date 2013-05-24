using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using GraphicsEngine.Core;
using SharpDX.Direct3D11;
using SharpDX.WIC;

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
                newTex.texture2D = LoadFromFile(Engine.g_device, Engine.g_image_factory, Path);
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


        #region Load utils

        public static Texture2D LoadFromFile(Device device, ImagingFactory2 factory, string fileName)
        {
            var bs = LoadBitmap(factory, fileName);
            var texture = CreateTexture2DFromBitmap(device, bs);
            return texture;

        }
        public static BitmapSource LoadBitmap(ImagingFactory2 factory, string filename)
        {
            var bitmapDecoder = new SharpDX.WIC.BitmapDecoder(
                factory,
                filename,
                SharpDX.WIC.DecodeOptions.CacheOnDemand
                );

            var result = new SharpDX.WIC.FormatConverter(factory);

            result.Initialize(
                bitmapDecoder.GetFrame(0),
                SharpDX.WIC.PixelFormat.Format32bppPRGBA,
                SharpDX.WIC.BitmapDitherType.None,
                null,
                0.0,
                SharpDX.WIC.BitmapPaletteType.Custom);

            return result;
        }

        public static Texture2D CreateTexture2DFromBitmap(Device device, BitmapSource bitmapSource)
        {
            // Allocate DataStream to receive the WIC image pixels
            int stride = bitmapSource.Size.Width * 4;
            using (var buffer = new SharpDX.DataStream(bitmapSource.Size.Height * stride, true, true))
            {
                // Copy the content of the WIC to the buffer
                bitmapSource.CopyPixels(stride, buffer);
                return new SharpDX.Direct3D11.Texture2D(device, new SharpDX.Direct3D11.Texture2DDescription()
                {
                    Width = bitmapSource.Size.Width,
                    Height = bitmapSource.Size.Height,
                    ArraySize = 1,
                    BindFlags = SharpDX.Direct3D11.BindFlags.ShaderResource,
                    Usage = SharpDX.Direct3D11.ResourceUsage.Immutable,
                    CpuAccessFlags = SharpDX.Direct3D11.CpuAccessFlags.None,
                    Format = SharpDX.DXGI.Format.R8G8B8A8_UNorm,
                    MipLevels = 1,
                    OptionFlags = SharpDX.Direct3D11.ResourceOptionFlags.None,
                    SampleDescription = new SharpDX.DXGI.SampleDescription(1, 0),
                }, new SharpDX.DataRectangle(buffer.DataPointer, stride));
            }
        }

        #endregion
    }
}
