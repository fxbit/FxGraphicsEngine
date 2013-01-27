using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

using WeifenLuo.WinFormsUI.Docking;

namespace Tester
{
    public partial class ConsoleOutput : DockContent
    {
        public ConsoleOutput()
        {
            InitializeComponent();
        }



        #region Toolbar menu

        private void saveToolStripButton_Click(object sender, EventArgs e)
        {
            if (saveFileDialog1.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                richTextBox1.SaveFile(saveFileDialog1.FileName);
            }
        }

        private void cutToolStripButton_Click(object sender, EventArgs e)
        {
            richTextBox1.SelectAll();
            richTextBox1.Copy();
            richTextBox1.Clear();
            richTextBox1.Select(richTextBox1.TextLength, 0);
        }

        private void copyToolStripButton_Click(object sender, EventArgs e)
        {
            richTextBox1.SelectAll();
            richTextBox1.Copy();
            richTextBox1.Select(richTextBox1.TextLength, 0);
        }

        private void toolStripButton1_Click(object sender, EventArgs e)
        {
            richTextBox1.Clear();
        } 

        #endregion



        #region Write Commands
        
        public void Write(String str)
        {
            richTextBox1.Text += str;
        }

        public void WriteLine(String str)
        {
            richTextBox1.Text += str + "\r\n";
        } 

        #endregion


    }
}
