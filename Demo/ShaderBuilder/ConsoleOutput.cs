using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.IO;

using WeifenLuo.WinFormsUI.Docking;

namespace ShaderBuilder
{
    public partial class ConsoleOutput : DockContent
    {

        /// <summary>
        /// The form that is create this console
        /// </summary>
        private Form1 form;

        public ConsoleOutput(Form1 form)
        {
            InitializeComponent();

            this.form = form;
        }



        #region Toolbar menu

        private void saveToolStripButton_Click(object sender, EventArgs e)
        {
            if (saveFileDialog1.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                StreamWriter sw = new StreamWriter(saveFileDialog1.FileName);
                foreach (String str in listBox1.Items)
                {
                    sw.WriteLine(str);
                }
            }
        }

        private void cutToolStripButton_Click(object sender, EventArgs e)
        {
#if false
            richTextBox1.SelectAll();
            richTextBox1.Copy();
            richTextBox1.Clear();
            richTextBox1.Select(richTextBox1.TextLength, 0);
#endif
        }

        private void copyToolStripButton_Click(object sender, EventArgs e)
        {
#if false
            richTextBox1.SelectAll();
            richTextBox1.Copy();
            richTextBox1.Select(richTextBox1.TextLength, 0); 
#endif
        }

        private void toolStripButton1_Click(object sender, EventArgs e)
        {
            listBox1.Items.Clear();
        } 

        #endregion



        #region Write Commands
        
        public void Write(String str)
        {
            listBox1.Items.Add(str);
        }

        public void WriteLine(String str)
        {
            form.Invoke((MethodInvoker)(()=>{ listBox1.Items.Add(str); Application.DoEvents(); }));
        } 

        #endregion



        private void listBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (listBox1.SelectedItem != null)
            {
                String str = (String)listBox1.SelectedItem;
                if (str.StartsWith("Error:"))
                {
                    String message = str.Remove(0, 6);
                    String []splitMessage = message.Split(' ');
                    splitMessage = splitMessage[0].Split('(');

                    if (File.Exists(splitMessage[0]))
                    {
                        String shaderPath = splitMessage[0];

                        form.OpenShader(shaderPath);
                        
                    }

                }
            }
        }


    }
}
