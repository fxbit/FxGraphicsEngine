using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

using ScintillaNET;
using System.Globalization;
using WeifenLuo.WinFormsUI.Docking;
using System.IO;

namespace ShaderBuilder
{
    public partial class Form1 : Form
    {

        private int _newDocumentCount = 0;

        /// <summary>
        /// Form for debuging.
        /// You can print that console by call UIConsole.Write
        /// </summary>
        public static ConsoleOutput UIConsole = null;


        public Form1()
        {
            InitializeComponent();

            // add UI console
            UIConsole = new ConsoleOutput(this);
            UIConsole.Show(dockPanel1, DockState.DockBottom);
            outputWindowToolStripMenuItem.Checked = true;
        }

        private void toolStripButton1_Click(object sender, EventArgs e)
        {
            DocumentForm doc = new DocumentForm();

            doc.Text = String.Format(CultureInfo.CurrentCulture, "{0}{1}", "Untitled", ++_newDocumentCount);
            doc.Show(dockPanel1);
            //toolIncremental.Searcher.Scintilla = doc.Scintilla;
        }

        private void selectFilesToolStripMenuItem_Click(object sender, EventArgs e)
        {
            ShaderSelection sh = new ShaderSelection();
            sh.Show(dockPanel1);
        }


        #region Add Output window

        private void outputWindowToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (outputWindowToolStripMenuItem.Checked)
            {
                if (UIConsole != null)
                {
                    UIConsole.Hide();
                }

                outputWindowToolStripMenuItem.Checked = false;
            }
            else
            {
                if (UIConsole == null)
                {
                    UIConsole = new ConsoleOutput(this);

                    // add the viewport to the dock
                    UIConsole.Show(dockPanel1, DockState.DockBottom);
                }
                else
                {
                    // add the viewport to the dock
                    UIConsole.Show(dockPanel1, DockState.DockBottom);
                }

                outputWindowToolStripMenuItem.Checked = true;
            }
        }

        #endregion


        public void OpenShader(String filePath)
        {
            DocumentForm doc = new DocumentForm();
            doc.Scintilla.Margins.Margin0.Width = 35;
            doc.Scintilla.Whitespace.Mode = WhitespaceMode.Invisible;
            doc.Scintilla.Text = File.ReadAllText(filePath);
            doc.Scintilla.UndoRedo.EmptyUndoBuffer();
            doc.Scintilla.Modified = false;
            doc.Text = Path.GetFileName(filePath);
            doc.FilePath = filePath;
            doc.Show(dockPanel1);


            // Use a built-in lexer and configuration
            doc.IniLexer = false;
            doc.Scintilla.ConfigurationManager.Language = "cs";
            doc.Scintilla.Indentation.SmartIndentType = SmartIndent.CPP;

        }
    }
}
