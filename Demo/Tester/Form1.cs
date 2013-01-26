﻿using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using WeifenLuo.WinFormsUI.Docking;

namespace Tester
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void toolStripButton1_Click(object sender, EventArgs e)
        {
            dockPanel1.SuspendLayout(true);

            Viewport viewport = new Viewport(this);
            viewport.Show(dockPanel1, DockState.Document);

            dockPanel1.ResumeLayout(true, true);
        }
    }
}
