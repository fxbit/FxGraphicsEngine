using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using SlimDX;
using SlimDX.DXGI;
using SlimDX.Direct3D11;

// resolve conflict - DXGI.Device & Direct3D10.Device
using Device = SlimDX.Direct3D11.Device;
using Buffer = SlimDX.Direct3D11.Buffer;
using Effect = SlimDX.Direct3D11.Effect;
using EffectFlags = SlimDX.D3DCompiler.EffectFlags;
using GraphicsEngine.Managers;
using System.Drawing;
using System.ComponentModel;
using System.Windows.Forms;
using System.Drawing.Design;
using FXFramework;

namespace GraphicsEngine.Core.Shaders.Modules {
    internal class FilteredFileNameEditor : UITypeEditor {
        private OpenFileDialog ofd = new OpenFileDialog();
        public override UITypeEditorEditStyle GetEditStyle(
         ITypeDescriptorContext context)
        {
            return UITypeEditorEditStyle.Modal;
        }

        public override object EditValue(
         ITypeDescriptorContext context,
         IServiceProvider provider,
         object value)
        {
            if (value == null)
                value = "";

            ofd.FileName = value.ToString();
            ofd.Filter = "Picture File|*.jpg|All Files|*.*";
            if (ofd.ShowDialog() == DialogResult.OK) {
                return ofd.FileName;
            }
            return base.EditValue(context, provider, value);
        }
    }

    public class ShaderVariable_Texture : ShaderVariable_Base {

        public FXResourceVariable Texture_Variable;
        private Texture Tex;
        private String _FilePath;

        [Editor(typeof(FilteredFileNameEditor),
          typeof(System.Drawing.Design.UITypeEditor))]
        public String FilePath
        {
            get { return _FilePath; }
            set { if (System.IO.File.Exists(value)) { _FilePath = value; Tex = TextureManager.AddTexture(_FilePath); } }
        }


        public ShaderVariable_Texture(String VariableName)
        {
            this.VariableName = VariableName;
        }

        public override void Init( FXEffect m_effect, FXConstantBuffer m_cb )
        {
            // bind the local variables with the shader
            Texture_Variable = m_effect.GetResourceByName(VariableName + "_Texture");

            // set the default values
            if (Tex != null)
                Texture_Variable.SetResource(Tex.shaderResource);
        }

    }
}
