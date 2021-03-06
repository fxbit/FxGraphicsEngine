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

using FxMaths.Images;
using FxMaths.GUI;
using FxMaths.Vector;
using FxMaths.GMaps;
using FxMaths;
using FxMaths.Matrix;

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

        public static SerialInput UISerialInput = null;

        public static SerialCapture UISerialCapture = null;

        /// <summary>
        /// The class that simulate the people movments.
        /// </summary>
        public static PeopleSimulation peopleSimulation = null;


        public static FxMatrixF katopsi = null;
        public static FxMatrixF heatMap = null;

        #region Form
        public MainForm()
        {
            InitializeComponent();

            katopsi = FxMatrixF.Load("Katopsi.jpg", FxMaths.Matrix.ColorSpace.Grayscale);

            // init the console
            UIConsole = new ConsoleOutput();
            UIConsole.Show(dockPanel1, DockState.DockBottomAutoHide);
            consoleOutputToolStripMenuItem.Checked = true;

            // init the people over view
            UIPeopleOverview = new PeopleOverview(katopsi);
            UIPeopleOverview.Show(dockPanel1, DockState.Document);
            peopleOverviewToolStripMenuItem.Checked = true;

            // Init Serial debugiing
            UISerialInput = new SerialInput();
            UISerialInput.Show(dockPanel1, DockState.Document);
            

            // Init serial Capture menu
            UISerialCapture = new SerialCapture();
            UISerialCapture.Show(dockPanel1, DockState.Document);
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


        #endregion



        #region Simulator

        int refreshCount = 0;

        private void peopleRefreshCB(PeopleSimulation ps)
        {
            
            /* now we must update People Overview */
            if (UIPeopleOverview != null)
                UIPeopleOverview.PeopleUpdate(ps.PersonList);

            /* create a heet map */
            if (refreshCount < 10)
            {
                if (refreshCount == 0)
                {
                    heatMap = katopsi.Copy();
                }

                foreach(Person p in ps.PersonList)
                {
                    heatMap.DrawCircle(p.Position, 10.0f, 0.5f);
                }
            }
            else
            {
                refreshCount = 0;
            }

            refreshCount++;

        }

        private void toolStripButton1_Click(object sender, EventArgs e)
        {
            peopleSimulation = new PeopleSimulation(20, new FxVector2f(560,145), new FxVector2f(-1,0), katopsi);
            peopleSimulation.Start(peopleRefreshCB);
        } 

        #endregion



        private void MainForm_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (peopleSimulation != null)
                peopleSimulation.Stop();

            if (UISerialCapture != null)
                UISerialCapture.Stop();
        }
    }
}
