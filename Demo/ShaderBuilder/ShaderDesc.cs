
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using SharpDX.Direct3D;
using SharpDX.D3DCompiler;
using System.IO;
using System.Drawing.Design;
using System.ComponentModel;
using System.Windows.Forms.Design;

namespace ShaderBuilder
{
    public class ShaderDesc
    {

        #region Private Variables
        private String _ShaderPath;
        private List<String> _EntryPoint;
        private String _OutputPath;
        private String _IncludePath;
        private FeatureLevel _BuildLevel;
        private ShaderFlags _Flags; 
        #endregion



        #region Public Properties
        public String ShaderPath
        {
            get { return _ShaderPath; }
            set { _ShaderPath = value; }
        }

        public List<String> EntryPoint
        {
            get { return _EntryPoint; }
            set { _EntryPoint = value; }
        }

        public String OutputPath
        {
            get { return _OutputPath; }
            set { _OutputPath = value; }
        }

        public String IncludePath
        {
            get { return _IncludePath; }
            set { _IncludePath = value; }
        }

        public FeatureLevel BuildLevel
        {
            get { return _BuildLevel; }
            set { _BuildLevel = value; }
        }

        [Editor(typeof(Utils.FlagEnumUIEditor),
        typeof(System.Drawing.Design.UITypeEditor))]
        public ShaderFlags Flags
        {
            get { return _Flags; }
            set { _Flags = value; }
        } 
        #endregion


        public ShaderDesc()
        {
            _EntryPoint = new List<string>();
            _EntryPoint.Add("main");
            _BuildLevel = FeatureLevel.Level_11_0;
            _Flags = ShaderFlags.OptimizationLevel3 | ShaderFlags.Debug;
            _IncludePath = null;
        }

        public override string ToString()
        {
            return Path.GetFileName(this._ShaderPath);
        }

        internal void Save(StringBuilder str)
        {
            // save the internal variables in string form
            str.AppendLine("ShaderPath=" + _ShaderPath);
            str.AppendLine("EntryPoint=" + _EntryPoint.Count);
            foreach (String entryPoint in _EntryPoint)
            {
                str.AppendLine(entryPoint);
            }

            str.AppendLine("OutputPath=" + _OutputPath);
            str.AppendLine("IncludePath=" + _IncludePath);
            str.AppendLine("BuildLevel=" + _BuildLevel.ToString());
            str.AppendLine("Flags=" + _Flags.ToString());
        }

        internal int Open(string[] fileLines, int p)
        {
            int  i=0;
            while( fileLines.Length > p+i && !fileLines[p+i].StartsWith("Shader:")){
                String line = fileLines[p+i];
                String []line_splited = line.Split('=');
                switch (line_splited[0])
                {
                    case "ShaderPath":
                        _ShaderPath = line_splited[1];
                        break;
                    case "EntryPoint":
                        _EntryPoint = new List<string>();
                        int numEntryPoint = int.Parse(line_splited[1]);
                        for (int j = 0; j < numEntryPoint; j++)
                        {
                          _EntryPoint.Add(fileLines[p + i + j + 1]);
                        }
                        i += numEntryPoint;
                        break;
                    case "OutputPath":
                        _OutputPath = line_splited[1];
                        break;
                    case "IncludePath":
                        _IncludePath = line_splited[1];
                        break;
                }

                i++;
            }

            this.BuildLevel = FeatureLevel.Level_11_0;
            this.Flags = ShaderFlags.Debug | ShaderFlags.OptimizationLevel3;
            
            return i-1;
        }
    }

}
