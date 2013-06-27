namespace Tester
{
    partial class TesterForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            WeifenLuo.WinFormsUI.Docking.DockPanelSkin dockPanelSkin1 = new WeifenLuo.WinFormsUI.Docking.DockPanelSkin();
            WeifenLuo.WinFormsUI.Docking.AutoHideStripSkin autoHideStripSkin1 = new WeifenLuo.WinFormsUI.Docking.AutoHideStripSkin();
            WeifenLuo.WinFormsUI.Docking.DockPanelGradient dockPanelGradient1 = new WeifenLuo.WinFormsUI.Docking.DockPanelGradient();
            WeifenLuo.WinFormsUI.Docking.TabGradient tabGradient1 = new WeifenLuo.WinFormsUI.Docking.TabGradient();
            WeifenLuo.WinFormsUI.Docking.DockPaneStripSkin dockPaneStripSkin1 = new WeifenLuo.WinFormsUI.Docking.DockPaneStripSkin();
            WeifenLuo.WinFormsUI.Docking.DockPaneStripGradient dockPaneStripGradient1 = new WeifenLuo.WinFormsUI.Docking.DockPaneStripGradient();
            WeifenLuo.WinFormsUI.Docking.TabGradient tabGradient2 = new WeifenLuo.WinFormsUI.Docking.TabGradient();
            WeifenLuo.WinFormsUI.Docking.DockPanelGradient dockPanelGradient2 = new WeifenLuo.WinFormsUI.Docking.DockPanelGradient();
            WeifenLuo.WinFormsUI.Docking.TabGradient tabGradient3 = new WeifenLuo.WinFormsUI.Docking.TabGradient();
            WeifenLuo.WinFormsUI.Docking.DockPaneStripToolWindowGradient dockPaneStripToolWindowGradient1 = new WeifenLuo.WinFormsUI.Docking.DockPaneStripToolWindowGradient();
            WeifenLuo.WinFormsUI.Docking.TabGradient tabGradient4 = new WeifenLuo.WinFormsUI.Docking.TabGradient();
            WeifenLuo.WinFormsUI.Docking.TabGradient tabGradient5 = new WeifenLuo.WinFormsUI.Docking.TabGradient();
            WeifenLuo.WinFormsUI.Docking.DockPanelGradient dockPanelGradient3 = new WeifenLuo.WinFormsUI.Docking.DockPanelGradient();
            WeifenLuo.WinFormsUI.Docking.TabGradient tabGradient6 = new WeifenLuo.WinFormsUI.Docking.TabGradient();
            WeifenLuo.WinFormsUI.Docking.TabGradient tabGradient7 = new WeifenLuo.WinFormsUI.Docking.TabGradient();
            this.statusStrip1 = new System.Windows.Forms.StatusStrip();
            this.toolStripStatusLabel_fps = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripStatusLabel2 = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripStatusLabel_triangles = new System.Windows.Forms.ToolStripStatusLabel();
            this.menuStrip1 = new System.Windows.Forms.MenuStrip();
            this.viewToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.addViewportToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.add2DViewportToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.outputToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.meshToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.addSamplePlaneToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.addVideoPlaneToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.addVideoSphereToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.addCEIDToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.direct2DToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.addImageToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.directComputeToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.delaunay2DToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.timer_statistics_update = new System.Windows.Forms.Timer(this.components);
            this.dockPanel1 = new WeifenLuo.WinFormsUI.Docking.DockPanel();
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            this.HitTest = new System.Windows.Forms.Timer(this.components);
            this.addCameraViewportToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.statusStrip1.SuspendLayout();
            this.menuStrip1.SuspendLayout();
            this.SuspendLayout();
            // 
            // statusStrip1
            // 
            this.statusStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripStatusLabel_fps,
            this.toolStripStatusLabel2,
            this.toolStripStatusLabel_triangles});
            this.statusStrip1.Location = new System.Drawing.Point(0, 404);
            this.statusStrip1.Name = "statusStrip1";
            this.statusStrip1.Padding = new System.Windows.Forms.Padding(1, 0, 10, 0);
            this.statusStrip1.Size = new System.Drawing.Size(663, 22);
            this.statusStrip1.TabIndex = 2;
            this.statusStrip1.Text = "statusStrip1";
            // 
            // toolStripStatusLabel_fps
            // 
            this.toolStripStatusLabel_fps.Name = "toolStripStatusLabel_fps";
            this.toolStripStatusLabel_fps.Size = new System.Drawing.Size(38, 17);
            this.toolStripStatusLabel_fps.Text = "FPS: 0";
            // 
            // toolStripStatusLabel2
            // 
            this.toolStripStatusLabel2.Name = "toolStripStatusLabel2";
            this.toolStripStatusLabel2.Size = new System.Drawing.Size(10, 17);
            this.toolStripStatusLabel2.Text = "|";
            // 
            // toolStripStatusLabel_triangles
            // 
            this.toolStripStatusLabel_triangles.Name = "toolStripStatusLabel_triangles";
            this.toolStripStatusLabel_triangles.Size = new System.Drawing.Size(67, 17);
            this.toolStripStatusLabel_triangles.Text = "Triangles: 0";
            // 
            // menuStrip1
            // 
            this.menuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.viewToolStripMenuItem,
            this.meshToolStripMenuItem,
            this.direct2DToolStripMenuItem,
            this.directComputeToolStripMenuItem});
            this.menuStrip1.Location = new System.Drawing.Point(0, 0);
            this.menuStrip1.Name = "menuStrip1";
            this.menuStrip1.Padding = new System.Windows.Forms.Padding(4, 2, 0, 2);
            this.menuStrip1.Size = new System.Drawing.Size(663, 24);
            this.menuStrip1.TabIndex = 4;
            this.menuStrip1.Text = "menuStrip1";
            // 
            // viewToolStripMenuItem
            // 
            this.viewToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.addViewportToolStripMenuItem,
            this.add2DViewportToolStripMenuItem,
            this.outputToolStripMenuItem,
            this.addCameraViewportToolStripMenuItem});
            this.viewToolStripMenuItem.Name = "viewToolStripMenuItem";
            this.viewToolStripMenuItem.Size = new System.Drawing.Size(44, 20);
            this.viewToolStripMenuItem.Text = "View";
            // 
            // addViewportToolStripMenuItem
            // 
            this.addViewportToolStripMenuItem.Name = "addViewportToolStripMenuItem";
            this.addViewportToolStripMenuItem.Size = new System.Drawing.Size(190, 22);
            this.addViewportToolStripMenuItem.Text = "Add 3D Viewport";
            this.addViewportToolStripMenuItem.Click += new System.EventHandler(this.addViewportToolStripMenuItem_Click);
            // 
            // add2DViewportToolStripMenuItem
            // 
            this.add2DViewportToolStripMenuItem.Name = "add2DViewportToolStripMenuItem";
            this.add2DViewportToolStripMenuItem.Size = new System.Drawing.Size(190, 22);
            this.add2DViewportToolStripMenuItem.Text = "Add 2D Viewport";
            this.add2DViewportToolStripMenuItem.Click += new System.EventHandler(this.add2DViewportToolStripMenuItem_Click);
            // 
            // outputToolStripMenuItem
            // 
            this.outputToolStripMenuItem.Name = "outputToolStripMenuItem";
            this.outputToolStripMenuItem.Size = new System.Drawing.Size(190, 22);
            this.outputToolStripMenuItem.Text = "Output";
            this.outputToolStripMenuItem.Click += new System.EventHandler(this.outputToolStripMenuItem_Click);
            // 
            // meshToolStripMenuItem
            // 
            this.meshToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.addSamplePlaneToolStripMenuItem,
            this.addVideoPlaneToolStripMenuItem,
            this.addVideoSphereToolStripMenuItem,
            this.addCEIDToolStripMenuItem});
            this.meshToolStripMenuItem.Name = "meshToolStripMenuItem";
            this.meshToolStripMenuItem.Size = new System.Drawing.Size(74, 20);
            this.meshToolStripMenuItem.Text = "DirectX 3D";
            // 
            // addSamplePlaneToolStripMenuItem
            // 
            this.addSamplePlaneToolStripMenuItem.Name = "addSamplePlaneToolStripMenuItem";
            this.addSamplePlaneToolStripMenuItem.Size = new System.Drawing.Size(170, 22);
            this.addSamplePlaneToolStripMenuItem.Text = "Add Sample Plane";
            this.addSamplePlaneToolStripMenuItem.Click += new System.EventHandler(this.addSamplePlaneToolStripMenuItem_Click);
            // 
            // addVideoPlaneToolStripMenuItem
            // 
            this.addVideoPlaneToolStripMenuItem.Name = "addVideoPlaneToolStripMenuItem";
            this.addVideoPlaneToolStripMenuItem.Size = new System.Drawing.Size(170, 22);
            this.addVideoPlaneToolStripMenuItem.Text = "Add Video Plane";
            this.addVideoPlaneToolStripMenuItem.Click += new System.EventHandler(this.addVideoPlaneToolStripMenuItem_Click);
            // 
            // addVideoSphereToolStripMenuItem
            // 
            this.addVideoSphereToolStripMenuItem.Name = "addVideoSphereToolStripMenuItem";
            this.addVideoSphereToolStripMenuItem.Size = new System.Drawing.Size(170, 22);
            this.addVideoSphereToolStripMenuItem.Text = "Add Video Sphere";
            this.addVideoSphereToolStripMenuItem.Click += new System.EventHandler(this.addVideoSphereToolStripMenuItem_Click);
            // 
            // addCEIDToolStripMenuItem
            // 
            this.addCEIDToolStripMenuItem.Name = "addCEIDToolStripMenuItem";
            this.addCEIDToolStripMenuItem.Size = new System.Drawing.Size(170, 22);
            this.addCEIDToolStripMenuItem.Text = "Add CEID";
            this.addCEIDToolStripMenuItem.Click += new System.EventHandler(this.addCEIDToolStripMenuItem_Click);
            // 
            // direct2DToolStripMenuItem
            // 
            this.direct2DToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.addImageToolStripMenuItem});
            this.direct2DToolStripMenuItem.Name = "direct2DToolStripMenuItem";
            this.direct2DToolStripMenuItem.Size = new System.Drawing.Size(74, 20);
            this.direct2DToolStripMenuItem.Text = "DirectX 2D";
            // 
            // addImageToolStripMenuItem
            // 
            this.addImageToolStripMenuItem.Name = "addImageToolStripMenuItem";
            this.addImageToolStripMenuItem.Size = new System.Drawing.Size(132, 22);
            this.addImageToolStripMenuItem.Text = "Add Image";
            this.addImageToolStripMenuItem.Click += new System.EventHandler(this.addImageToolStripMenuItem_Click);
            // 
            // directComputeToolStripMenuItem
            // 
            this.directComputeToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.delaunay2DToolStripMenuItem});
            this.directComputeToolStripMenuItem.Name = "directComputeToolStripMenuItem";
            this.directComputeToolStripMenuItem.Size = new System.Drawing.Size(100, 20);
            this.directComputeToolStripMenuItem.Text = "DirectCompute";
            // 
            // delaunay2DToolStripMenuItem
            // 
            this.delaunay2DToolStripMenuItem.Name = "delaunay2DToolStripMenuItem";
            this.delaunay2DToolStripMenuItem.Size = new System.Drawing.Size(137, 22);
            this.delaunay2DToolStripMenuItem.Text = "Delaunay2D";
            this.delaunay2DToolStripMenuItem.Click += new System.EventHandler(this.delaunay2DToolStripMenuItem_Click);
            // 
            // timer_statistics_update
            // 
            this.timer_statistics_update.Enabled = true;
            this.timer_statistics_update.Tick += new System.EventHandler(this.timer_statistics_update_Tick);
            // 
            // dockPanel1
            // 
            this.dockPanel1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.dockPanel1.DocumentStyle = WeifenLuo.WinFormsUI.Docking.DocumentStyle.DockingWindow;
            this.dockPanel1.Location = new System.Drawing.Point(0, 24);
            this.dockPanel1.Margin = new System.Windows.Forms.Padding(2);
            this.dockPanel1.Name = "dockPanel1";
            this.dockPanel1.Size = new System.Drawing.Size(663, 380);
            dockPanelGradient1.EndColor = System.Drawing.SystemColors.ControlLight;
            dockPanelGradient1.StartColor = System.Drawing.SystemColors.ControlLight;
            autoHideStripSkin1.DockStripGradient = dockPanelGradient1;
            tabGradient1.EndColor = System.Drawing.SystemColors.Control;
            tabGradient1.StartColor = System.Drawing.SystemColors.Control;
            tabGradient1.TextColor = System.Drawing.SystemColors.ControlDarkDark;
            autoHideStripSkin1.TabGradient = tabGradient1;
            autoHideStripSkin1.TextFont = new System.Drawing.Font("Segoe UI", 9F);
            dockPanelSkin1.AutoHideStripSkin = autoHideStripSkin1;
            tabGradient2.EndColor = System.Drawing.SystemColors.ControlLightLight;
            tabGradient2.StartColor = System.Drawing.SystemColors.ControlLightLight;
            tabGradient2.TextColor = System.Drawing.SystemColors.ControlText;
            dockPaneStripGradient1.ActiveTabGradient = tabGradient2;
            dockPanelGradient2.EndColor = System.Drawing.SystemColors.Control;
            dockPanelGradient2.StartColor = System.Drawing.SystemColors.Control;
            dockPaneStripGradient1.DockStripGradient = dockPanelGradient2;
            tabGradient3.EndColor = System.Drawing.SystemColors.ControlLight;
            tabGradient3.StartColor = System.Drawing.SystemColors.ControlLight;
            tabGradient3.TextColor = System.Drawing.SystemColors.ControlText;
            dockPaneStripGradient1.InactiveTabGradient = tabGradient3;
            dockPaneStripSkin1.DocumentGradient = dockPaneStripGradient1;
            dockPaneStripSkin1.TextFont = new System.Drawing.Font("Segoe UI", 9F);
            tabGradient4.EndColor = System.Drawing.SystemColors.ActiveCaption;
            tabGradient4.LinearGradientMode = System.Drawing.Drawing2D.LinearGradientMode.Vertical;
            tabGradient4.StartColor = System.Drawing.SystemColors.GradientActiveCaption;
            tabGradient4.TextColor = System.Drawing.SystemColors.ActiveCaptionText;
            dockPaneStripToolWindowGradient1.ActiveCaptionGradient = tabGradient4;
            tabGradient5.EndColor = System.Drawing.SystemColors.Control;
            tabGradient5.StartColor = System.Drawing.SystemColors.Control;
            tabGradient5.TextColor = System.Drawing.SystemColors.ControlText;
            dockPaneStripToolWindowGradient1.ActiveTabGradient = tabGradient5;
            dockPanelGradient3.EndColor = System.Drawing.SystemColors.ControlLight;
            dockPanelGradient3.StartColor = System.Drawing.SystemColors.ControlLight;
            dockPaneStripToolWindowGradient1.DockStripGradient = dockPanelGradient3;
            tabGradient6.EndColor = System.Drawing.SystemColors.InactiveCaption;
            tabGradient6.LinearGradientMode = System.Drawing.Drawing2D.LinearGradientMode.Vertical;
            tabGradient6.StartColor = System.Drawing.SystemColors.GradientInactiveCaption;
            tabGradient6.TextColor = System.Drawing.SystemColors.InactiveCaptionText;
            dockPaneStripToolWindowGradient1.InactiveCaptionGradient = tabGradient6;
            tabGradient7.EndColor = System.Drawing.Color.Transparent;
            tabGradient7.StartColor = System.Drawing.Color.Transparent;
            tabGradient7.TextColor = System.Drawing.SystemColors.ControlDarkDark;
            dockPaneStripToolWindowGradient1.InactiveTabGradient = tabGradient7;
            dockPaneStripSkin1.ToolWindowGradient = dockPaneStripToolWindowGradient1;
            dockPanelSkin1.DockPaneStripSkin = dockPaneStripSkin1;
            this.dockPanel1.Skin = dockPanelSkin1;
            this.dockPanel1.TabIndex = 0;
            // 
            // openFileDialog1
            // 
            this.openFileDialog1.FileName = "openFileDialog1";
            // 
            // HitTest
            // 
            this.HitTest.Enabled = true;
            this.HitTest.Tick += new System.EventHandler(this.HitTest_Tick);
            // 
            // addCameraViewportToolStripMenuItem
            // 
            this.addCameraViewportToolStripMenuItem.Name = "addCameraViewportToolStripMenuItem";
            this.addCameraViewportToolStripMenuItem.Size = new System.Drawing.Size(190, 22);
            this.addCameraViewportToolStripMenuItem.Text = "Add Camera Viewport";
            this.addCameraViewportToolStripMenuItem.Click += new System.EventHandler(this.addCameraViewportToolStripMenuItem_Click);
            // 
            // TesterForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(663, 426);
            this.Controls.Add(this.dockPanel1);
            this.Controls.Add(this.statusStrip1);
            this.Controls.Add(this.menuStrip1);
            this.MainMenuStrip = this.menuStrip1;
            this.Margin = new System.Windows.Forms.Padding(2);
            this.Name = "TesterForm";
            this.Text = "Form1";
            this.ResizeEnd += new System.EventHandler(this.Form1_ResizeEnd);
            this.statusStrip1.ResumeLayout(false);
            this.statusStrip1.PerformLayout();
            this.menuStrip1.ResumeLayout(false);
            this.menuStrip1.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private WeifenLuo.WinFormsUI.Docking.DockPanel dockPanel1;
        private System.Windows.Forms.StatusStrip statusStrip1;
        private System.Windows.Forms.MenuStrip menuStrip1;
        private System.Windows.Forms.ToolStripMenuItem viewToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem addViewportToolStripMenuItem;
        private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel_fps;
        private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel2;
        private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel_triangles;
        private System.Windows.Forms.Timer timer_statistics_update;
        private System.Windows.Forms.ToolStripMenuItem meshToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem addSamplePlaneToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem add2DViewportToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem direct2DToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem addImageToolStripMenuItem;
        private System.Windows.Forms.OpenFileDialog openFileDialog1;
        private System.Windows.Forms.ToolStripMenuItem directComputeToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem delaunay2DToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem outputToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem addVideoPlaneToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem addVideoSphereToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem addCEIDToolStripMenuItem;
        private System.Windows.Forms.Timer HitTest;
        private System.Windows.Forms.ToolStripMenuItem addCameraViewportToolStripMenuItem;
    }
}

