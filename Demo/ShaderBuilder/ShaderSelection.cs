using GraphicsEngine.Core;
using SharpDX.D3DCompiler;
using SharpDX.Direct3D;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Design;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;

namespace ShaderBuilder
{
    public partial class ShaderSelection : DockContent
    {

        List<ShaderDesc> shader_list;
        List<String> include_paths;


        public ShaderSelection()
        {
            InitializeComponent();

            shader_list = new List<ShaderDesc>();
            include_paths = new List<String>();
        }



        #region Add Files For building


        private void toolStripButton4_Click(object sender, EventArgs e)
        {
            openFileDialog1.Multiselect = true;
            openFileDialog1.Filter = "HLSL Files (*.hlsl)|*.hlsl|All Files (*.*)|*.*";
            if (openFileDialog1.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                foreach (String str in openFileDialog1.FileNames)
                {
                    ShaderDesc desc = new ShaderDesc();
                    desc.ShaderPath = str;
                    listBox1.Items.Add(desc);
                    shader_list.Add(desc);
                }
            }
        } 

        #endregion



        #region Building Shaders

        private void toolStripButton3_Click(object sender, EventArgs e)
        {
            List<String> logs = new List<string>();
            //Parallel.ForEach(shader_list, shader =>
            foreach (ShaderDesc shader in shader_list)
            {

                WriteLog("Start Building :" + shader.ToString());
                logs.Add("Start Building :" + shader.ToString());

                // init the include class and subpath
                IncludeFX includeFX = new IncludeFX();
                if (shader.IncludePath == null)
                {
                    includeFX.IncludeDirectory = System.Windows.Forms.Application.StartupPath + "\\ComputeHLSL\\";
                }
                else
                {
                    includeFX.IncludeDirectory = shader.IncludePath;
                }

                // init the include class and subpath
                ShaderBytecode bytecode = null;

                // set the compile level base on running Feature level.
                String CompileLevelCS;
                if (shader.BuildLevel == FeatureLevel.Level_11_0)
                {
                    CompileLevelCS = "cs_5_0";
                }
                else
                {
                    CompileLevelCS = "cs_4_0";
                }

                foreach (String entryPoint in shader.EntryPoint)
                {
                    try
                    {
                        /// compile the shader to byte code
                        bytecode = ShaderBytecode.CompileFromFile(
                                                    shader.ShaderPath,               /// File Path of the file containing the code 
                                                    entryPoint,               /// The name of the executable function
                                                    CompileLevelCS,                  /// What specifications (shader version) to compile with cs_4_0 for directX10 and cs_5_0 for directx11
                                                    shader.Flags, EffectFlags.None, null, includeFX);
                    }
                    catch (Exception ex)
                    {
                        WriteLog("Error:" + ex.Message);
                        logs.Add("Error:" + ex.Message);
                        break;
                    }

                    WriteLog("Entry Point:" + entryPoint + " builded");
                    logs.Add("Entry Point:" + entryPoint + " builded");
                    bytecode.Save(shader.OutputPath + "\\" + Path.GetFileNameWithoutExtension(shader.ShaderPath) + "." + entryPoint + ".fxo");
                }
                WriteLog("Building Completed of" + shader.ToString());
                logs.Add("Building Completed of" + shader.ToString());
                //});
            }
            //foreach (String str in logs)
              //  WriteLog(str);
        } 

        #endregion


        #region Listbox Events



        #region Selection Changing
        private void listBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (listBox1.SelectedItem != null)
            {
                propertyGrid1.SelectedObject = listBox1.SelectedItem;
            }
        }  
        #endregion


        #endregion



        private void WriteLog(String str)
        {
            Form1.UIConsole.WriteLine(str);
        }





        private void toolStripButton1_Click(object sender, EventArgs e)
        {
            if (openFileDialog1.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                StreamReader sr = new StreamReader(openFileDialog1.FileName);
                String file = sr.ReadToEnd();
                sr.Close();

                char []splitChar = {'\r','\n'};
                String []fileLines = file.Split(splitChar, StringSplitOptions.RemoveEmptyEntries);
                for(int i=0; i<fileLines.Length;i++)
                {
                    if (fileLines[i].StartsWith("Shader:"))
                    {
                        ShaderDesc desc = new ShaderDesc();
                        i += desc.Open(fileLines, i+1);
                        shader_list.Add(desc);
                    }
                }

                listBox1.Items.AddRange(shader_list.ToArray());
            }
        }

        private void toolStripButton2_Click(object sender, EventArgs e)
        {
            if (saveFileDialog1.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                StringBuilder str = new StringBuilder();
                int count = 0;
                foreach (ShaderDesc desc in shader_list)
                {
                    str.AppendLine("Shader:" + count.ToString());
                    desc.Save(str);
                    count++;
                }
                StreamWriter sw = new StreamWriter(saveFileDialog1.FileName);
                sw.Write(str.ToString());
                sw.Close();
            }
        }


    }

    
}
