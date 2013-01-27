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
            WeifenLuo.WinFormsUI.Docking.DockPanelSkin dockPanelSkin3 = new WeifenLuo.WinFormsUI.Docking.DockPanelSkin();
            WeifenLuo.WinFormsUI.Docking.AutoHideStripSkin autoHideStripSkin3 = new WeifenLuo.WinFormsUI.Docking.AutoHideStripSkin();
            WeifenLuo.WinFormsUI.Docking.DockPanelGradient dockPanelGradient7 = new WeifenLuo.WinFormsUI.Docking.DockPanelGradient();
            WeifenLuo.WinFormsUI.Docking.TabGradient tabGradient15 = new WeifenLuo.WinFormsUI.Docking.TabGradient();
            WeifenLuo.WinFormsUI.Docking.DockPaneStripSkin dockPaneStripSkin3 = new WeifenLuo.WinFormsUI.Docking.DockPaneStripSkin();
            WeifenLuo.WinFormsUI.Docking.DockPaneStripGradient dockPaneStripGradient3 = new WeifenLuo.WinFormsUI.Docking.DockPaneStripGradient();
            WeifenLuo.WinFormsUI.Docking.TabGradient tabGradient16 = new WeifenLuo.WinFormsUI.Docking.TabGradient();
            WeifenLuo.WinFormsUI.Docking.DockPanelGradient dockPanelGradient8 = new WeifenLuo.WinFormsUI.Docking.DockPanelGradient();
            WeifenLuo.WinFormsUI.Docking.TabGradient tabGradient17 = new WeifenLuo.WinFormsUI.Docking.TabGradient();
            WeifenLuo.WinFormsUI.Docking.DockPaneStripToolWindowGradient dockPaneStripToolWindowGradient3 = new WeifenLuo.WinFormsUI.Docking.DockPaneStripToolWindowGradient();
            WeifenLuo.WinFormsUI.Docking.TabGradient tabGradient18 = new WeifenLuo.WinFormsUI.Docking.TabGradient();
            WeifenLuo.WinFormsUI.Docking.TabGradient tabGradient19 = new WeifenLuo.WinFormsUI.Docking.TabGradient();
            WeifenLuo.WinFormsUI.Docking.DockPanelGradient dockPanelGradient9 = new WeifenLuo.WinFormsUI.Docking.DockPanelGradient();
            WeifenLuo.WinFormsUI.Docking.TabGradient tabGradient20 = new WeifenLuo.WinFormsUI.Docking.TabGradient();
            WeifenLuo.WinFormsUI.Docking.TabGradient tabGradient21 = new WeifenLuo.WinFormsUI.Docking.TabGradient();
            this.statusStrip1 = new System.Windows.Forms.StatusStrip();
            this.toolStripStatusLabel_fps = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripStatusLabel2 = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripStatusLabel_triangles = new System.Windows.Forms.ToolStripStatusLabel();
            this.menuStrip1 = new System.Windows.Forms.MenuStrip();
            this.viewToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.addViewportToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.add2DViewportToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.meshToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.addSamplePlaneToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.direct2DToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.addImageToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.timer_statistics_update = new System.Windows.Forms.Timer(this.components);
            this.dockPanel1 = new WeifenLuo.WinFormsUI.Docking.DockPanel();
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            this.directComputeToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.delaunay2DToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.outputToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
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
            this.statusStrip1.Location = new System.Drawing.Point(0, 499);
            this.statusStrip1.Name = "statusStrip1";
            this.statusStrip1.Size = new System.Drawing.Size(884, 25);
            this.statusStrip1.TabIndex = 2;
            this.statusStrip1.Text = "statusStrip1";
            // 
            // toolStripStatusLabel_fps
            // 
            this.toolStripStatusLabel_fps.Name = "toolStripStatusLabel_fps";
            this.toolStripStatusLabel_fps.Size = new System.Drawing.Size(47, 20);
            this.toolStripStatusLabel_fps.Text = "FPS: 0";
            // 
            // toolStripStatusLabel2
            // 
            this.toolStripStatusLabel2.Name = "toolStripStatusLabel2";
            this.toolStripStatusLabel2.Size = new System.Drawing.Size(13, 20);
            this.toolStripStatusLabel2.Text = "|";
            // 
            // toolStripStatusLabel_triangles
            // 
            this.toolStripStatusLabel_triangles.Name = "toolStripStatusLabel_triangles";
            this.toolStripStatusLabel_triangles.Size = new System.Drawing.Size(84, 20);
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
            this.menuStrip1.Size = new System.Drawing.Size(884, 28);
            this.menuStrip1.TabIndex = 4;
            this.menuStrip1.Text = "menuStrip1";
            // 
            // viewToolStripMenuItem
            // 
            this.viewToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.addViewportToolStripMenuItem,
            this.add2DViewportToolStripMenuItem,
            this.outputToolStripMenuItem});
            this.viewToolStripMenuItem.Name = "viewToolStripMenuItem";
            this.viewToolStripMenuItem.Size = new System.Drawing.Size(53, 24);
            this.viewToolStripMenuItem.Text = "View";
            // 
            // addViewportToolStripMenuItem
            // 
            this.addViewportToolStripMenuItem.Name = "addViewportToolStripMenuItem";
            this.addViewportToolStripMenuItem.Size = new System.Drawing.Size(193, 24);
            this.addViewportToolStripMenuItem.Text = "Add 3D Viewport";
            this.addViewportToolStripMenuItem.Click += new System.EventHandler(this.addViewportToolStripMenuItem_Click);
            // 
            // add2DViewportToolStripMenuItem
            // 
            this.add2DViewportToolStripMenuItem.Name = "add2DViewportToolStripMenuItem";
            this.add2DViewportToolStripMenuItem.Size = new System.Drawing.Size(193, 24);
            this.add2DViewportToolStripMenuItem.Text = "Add 2D Viewport";
            this.add2DViewportToolStripMenuItem.Click += new System.EventHandler(this.add2DViewportToolStripMenuItem_Click);
            // 
            // meshToolStripMenuItem
            // 
            this.meshToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.addSamplePlaneToolStripMenuItem});
            this.meshToolStripMenuItem.Name = "meshToolStripMenuItem";
            this.meshToolStripMenuItem.Size = new System.Drawing.Size(93, 24);
            this.meshToolStripMenuItem.Text = "DirectX 3D";
            // 
            // addSamplePlaneToolStripMenuItem
            // 
            this.addSamplePlaneToolStripMenuItem.Name = "addSamplePlaneToolStripMenuItem";
            this.addSamplePlaneToolStripMenuItem.Size = new System.Drawing.Size(200, 24);
            this.addSamplePlaneToolStripMenuItem.Text = "Add Sample Plane";
            this.addSamplePlaneToolStripMenuItem.Click += new System.EventHandler(this.addSamplePlaneToolStripMenuItem_Click);
            // 
            // direct2DToolStripMenuItem
            // 
            this.direct2DToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.addImageToolStripMenuItem});
            this.direct2DToolStripMenuItem.Name = "direct2DToolStripMenuItem";
            this.direct2DToolStripMenuItem.Size = new System.Drawing.Size(93, 24);
            this.direct2DToolStripMenuItem.Text = "DirectX 2D";
            // 
            // addImageToolStripMenuItem
            // 
            this.addImageToolStripMenuItem.Name = "addImageToolStripMenuItem";
            this.addImageToolStripMenuItem.Size = new System.Drawing.Size(152, 24);
            this.addImageToolStripMenuItem.Text = "Add Image";
            this.addImageToolStripMenuItem.Click += new System.EventHandler(this.addImageToolStripMenuItem_Click);
            // 
            // timer_statistics_update
            // 
            this.timer_statistics_update.Enabled = true;
            this.timer_statistics_update.Interval = 1000;
            this.timer_statistics_update.Tick += new System.EventHandler(this.timer_statistics_update_Tick);
            // 
            // dockPanel1
            // 
            this.dockPanel1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.dockPanel1.DocumentStyle = WeifenLuo.WinFormsUI.Docking.DocumentStyle.DockingWindow;
            this.dockPanel1.Location = new System.Drawing.Point(0, 28);
            this.dockPanel1.Name = "dockPanel1";
            this.dockPanel1.Size = new System.Drawing.Size(884, 471);
            dockPanelGradient7.EndColor = System.Drawing.SystemColors.ControlLight;
            dockPanelGradient7.StartColor = System.Drawing.SystemColors.ControlLight;
            autoHideStripSkin3.DockStripGradient = dockPanelGradient7;
            tabGradient15.EndColor = System.Drawing.SystemColors.Control;
            tabGradient15.StartColor = System.Drawing.SystemColors.Control;
            tabGradient15.TextColor = System.Drawing.SystemColors.ControlDarkDark;
            autoHideStripSkin3.TabGradient = tabGradient15;
            autoHideStripSkin3.TextFont = new System.Drawing.Font("Segoe UI", 9F);
            dockPanelSkin3.AutoHideStripSkin = autoHideStripSkin3;
            tabGradient16.EndColor = System.Drawing.SystemColors.ControlLightLight;
            tabGradient16.StartColor = System.Drawing.SystemColors.ControlLightLight;
            tabGradient16.TextColor = System.Drawing.SystemColors.ControlText;
            dockPaneStripGradient3.ActiveTabGradient = tabGradient16;
            dockPanelGradient8.EndColor = System.Drawing.SystemColors.Control;
            dockPanelGradient8.StartColor = System.Drawing.SystemColors.Control;
            dockPaneStripGradient3.DockStripGradient = dockPanelGradient8;
            tabGradient17.EndColor = System.Drawing.SystemColors.ControlLight;
            tabGradient17.StartColor = System.Drawing.SystemColors.ControlLight;
            tabGradient17.TextColor = System.Drawing.SystemColors.ControlText;
            dockPaneStripGradient3.InactiveTabGradient = tabGradient17;
            dockPaneStripSkin3.DocumentGradient = dockPaneStripGradient3;
            dockPaneStripSkin3.TextFont = new System.Drawing.Font("Segoe UI", 9F);
            tabGradient18.EndColor = System.Drawing.SystemColors.ActiveCaption;
            tabGradient18.LinearGradientMode = System.Drawing.Drawing2D.LinearGradientMode.Vertical;
            tabGradient18.StartColor = System.Drawing.SystemColors.GradientActiveCaption;
            tabGradient18.TextColor = System.Drawing.SystemColors.ActiveCaptionText;
            dockPaneStripToolWindowGradient3.ActiveCaptionGradient = tabGradient18;
            tabGradient19.EndColor = System.Drawing.SystemColors.Control;
            tabGradient19.StartColor = System.Drawing.SystemColors.Control;
            tabGradient19.TextColor = System.Drawing.SystemColors.ControlText;
            dockPaneStripToolWindowGradient3.ActiveTabGradient = tabGradient19;
            dockPanelGradient9.EndColor = System.Drawing.SystemColors.ControlLight;
            dockPanelGradient9.StartColor = System.Drawing.SystemColors.ControlLight;
            dockPaneStripToolWindowGradient3.DockStripGradient = dockPanelGradient9;
            tabGradient20.EndColor = System.Drawing.SystemColors.InactiveCaption;
            tabGradient20.LinearGradientMode = System.Drawing.Drawing2D.LinearGradientMode.Vertical;
            tabGradient20.StartColor = System.Drawing.SystemColors.GradientInactiveCaption;
            tabGradient20.TextColor = System.Drawing.SystemColors.InactiveCaptionText;
            dockPaneStripToolWindowGradient3.InactiveCaptionGradient = tabGradient20;
            tabGradient21.EndColor = System.Drawing.Color.Transparent;
            tabGradient21.StartColor = System.Drawing.Color.Transparent;
            tabGradient21.TextColor = System.Drawing.SystemColors.ControlDarkDark;
            dockPaneStripToolWindowGradient3.InactiveTabGradient = tabGradient21;
            dockPaneStripSkin3.ToolWindowGradient = dockPaneStripToolWindowGradient3;
            dockPanelSkin3.DockPaneStripSkin = dockPaneStripSkin3;
            this.dockPanel1.Skin = dockPanelSkin3;
            this.dockPanel1.TabIndex = 0;
            // 
            // openFileDialog1
            // 
            this.openFileDialog1.FileName = "openFileDialog1";
            // 
            // directComputeToolStripMenuItem
            // 
            this.directComputeToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.delaunay2DToolStripMenuItem});
            this.directComputeToolStripMenuItem.Name = "directComputeToolStripMenuItem";
            this.directComputeToolStripMenuItem.Size = new System.Drawing.Size(122, 24);
            this.directComputeToolStripMenuItem.Text = "DirectCompute";
            // 
            // delaunay2DToolStripMenuItem
            // 
            this.delaunay2DToolStripMenuItem.Name = "delaunay2DToolStripMenuItem";
            this.delaunay2DToolStripMenuItem.Size = new System.Drawing.Size(159, 24);
            this.delaunay2DToolStripMenuItem.Text = "Delaunay2D";
            this.delaunay2DToolStripMenuItem.Click += new System.EventHandler(this.delaunay2DToolStripMenuItem_Click);
            // 
            // outputToolStripMenuItem
            // 
            this.outputToolStripMenuItem.Name = "outputToolStripMenuItem";
            this.outputToolStripMenuItem.Size = new System.Drawing.Size(193, 24);
            this.outputToolStripMenuItem.Text = "Output";
            this.outputToolStripMenuItem.Click += new System.EventHandler(this.outputToolStripMenuItem_Click);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(884, 524);
            this.Controls.Add(this.dockPanel1);
            this.Controls.Add(this.statusStrip1);
            this.Controls.Add(this.menuStrip1);
            this.MainMenuStrip = this.menuStrip1;
            this.Name = "Form1";
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
    }
}

