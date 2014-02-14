using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using WeifenLuo.WinFormsUI.Docking;

using FxMaths.Images;
using FxMaths.GUI;
using FxMaths.Vector;
using FxMaths.GMaps;
using FxMaths;

namespace MainForm
{
    public partial class MainForm : Form
    {
        /// <summary>
        /// Form for debuging.
        /// You can print that console by call UIConsole.Write
        /// </summary>
        public static ConsoleOutput UIConsole = null;

        /// <summary>
        /// Form for drawing people position.
        /// </summary>
        public static PeopleOverview UIPeopleOverview = null;



        #region Form
        public MainForm()
        {
            InitializeComponent();


            // init the console
            UIConsole = new ConsoleOutput();
            UIConsole.Show(dockPanel1, DockState.DockBottom);
            consoleOutputToolStripMenuItem.Checked = true;

            // init the people over view
            UIPeopleOverview = new PeopleOverview();
            UIPeopleOverview.Show(dockPanel1, DockState.Document);
            peopleOverviewToolStripMenuItem.Checked = true;
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void exitToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Application.Exit();
        } 
        #endregion




        #region Console Output window

        private void consoleOutputToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (consoleOutputToolStripMenuItem.Checked)
            {
                UIConsole.Hide();
            }
            else
            {
                if (UIConsole == null)
                    UIConsole = new ConsoleOutput();

                // add the viewport to the dock
                UIConsole.Show(dockPanel1, DockState.DockBottom);
            }

            consoleOutputToolStripMenuItem.Checked = !consoleOutputToolStripMenuItem.Checked;
        }

        #endregion




        #region People Overview


        private void peopleOverviewToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (peopleOverviewToolStripMenuItem.Checked)
            {
                UIPeopleOverview.Hide();
            }
            else
            {
                if (UIPeopleOverview == null)
                    UIPeopleOverview = new PeopleOverview();

                // add the viewport to the dock
                UIPeopleOverview.Show(dockPanel1, DockState.Document);
            }

            peopleOverviewToolStripMenuItem.Checked = !peopleOverviewToolStripMenuItem.Checked;
        }

        private void toolStripButton1_Click(object sender, EventArgs e)
        {

        } 


        #endregion



        #region Simulator
        
        #endregion
    }
}
