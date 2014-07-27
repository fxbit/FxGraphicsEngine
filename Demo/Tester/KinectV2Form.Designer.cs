namespace Tester
{
    partial class KinectV2Form
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(KinectV2Form));
            this.RenderArea = new System.Windows.Forms.PictureBox();
            this.toolStrip1 = new System.Windows.Forms.ToolStrip();
            this.toolStripButton1 = new System.Windows.Forms.ToolStripButton();
            this.toolStripLabel1 = new System.Windows.Forms.ToolStripLabel();
            this.toolStripLabel_fps = new System.Windows.Forms.ToolStripLabel();
            this.toolStripButton2 = new System.Windows.Forms.ToolStripButton();
            this.toolStripButton3 = new System.Windows.Forms.ToolStripButton();
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.tabPage1 = new System.Windows.Forms.TabPage();
            this.tabPage2 = new System.Windows.Forms.TabPage();
            this.canvas1 = new FxMaths.GUI.Canvas();
            this.tabPage3 = new System.Windows.Forms.TabPage();
            this.canvas_ellipse = new FxMaths.GUI.Canvas();
            this.toolStrip2 = new System.Windows.Forms.ToolStrip();
            this.toolStripButton_EllipseOpenImage = new System.Windows.Forms.ToolStripButton();
            this.toolStripButton_eclipse_SelectSubRegion = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator1 = new System.Windows.Forms.ToolStripSeparator();
            this.toolStripButton_ellipse_extract = new System.Windows.Forms.ToolStripButton();
            ((System.ComponentModel.ISupportInitialize)(this.RenderArea)).BeginInit();
            this.toolStrip1.SuspendLayout();
            this.tabControl1.SuspendLayout();
            this.tabPage1.SuspendLayout();
            this.tabPage2.SuspendLayout();
            this.tabPage3.SuspendLayout();
            this.toolStrip2.SuspendLayout();
            this.SuspendLayout();
            // 
            // RenderArea
            // 
            this.RenderArea.Dock = System.Windows.Forms.DockStyle.Fill;
            this.RenderArea.Location = new System.Drawing.Point(3, 3);
            this.RenderArea.Name = "RenderArea";
            this.RenderArea.Size = new System.Drawing.Size(727, 603);
            this.RenderArea.TabIndex = 0;
            this.RenderArea.TabStop = false;
            this.RenderArea.MouseClick += new System.Windows.Forms.MouseEventHandler(this.RenderArea_MouseClick);
            this.RenderArea.Resize += new System.EventHandler(this.RenderArea_Resize);
            // 
            // toolStrip1
            // 
            this.toolStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripButton1,
            this.toolStripLabel1,
            this.toolStripLabel_fps,
            this.toolStripButton2,
            this.toolStripButton3});
            this.toolStrip1.Location = new System.Drawing.Point(0, 0);
            this.toolStrip1.Name = "toolStrip1";
            this.toolStrip1.Size = new System.Drawing.Size(741, 25);
            this.toolStrip1.TabIndex = 1;
            this.toolStrip1.Text = "toolStrip1";
            // 
            // toolStripButton1
            // 
            this.toolStripButton1.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.toolStripButton1.Image = ((System.Drawing.Image)(resources.GetObject("toolStripButton1.Image")));
            this.toolStripButton1.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.toolStripButton1.Name = "toolStripButton1";
            this.toolStripButton1.Size = new System.Drawing.Size(23, 22);
            this.toolStripButton1.Text = "StartKinect";
            this.toolStripButton1.Click += new System.EventHandler(this.toolStripButton1_Click);
            // 
            // toolStripLabel1
            // 
            this.toolStripLabel1.Name = "toolStripLabel1";
            this.toolStripLabel1.Size = new System.Drawing.Size(29, 22);
            this.toolStripLabel1.Text = "FPS:";
            // 
            // toolStripLabel_fps
            // 
            this.toolStripLabel_fps.Name = "toolStripLabel_fps";
            this.toolStripLabel_fps.Size = new System.Drawing.Size(13, 22);
            this.toolStripLabel_fps.Text = "0";
            // 
            // toolStripButton2
            // 
            this.toolStripButton2.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.toolStripButton2.Image = ((System.Drawing.Image)(resources.GetObject("toolStripButton2.Image")));
            this.toolStripButton2.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.toolStripButton2.Name = "toolStripButton2";
            this.toolStripButton2.Size = new System.Drawing.Size(23, 22);
            this.toolStripButton2.Text = "GetOneFrame";
            this.toolStripButton2.Click += new System.EventHandler(this.toolStripButton2_Click);
            // 
            // toolStripButton3
            // 
            this.toolStripButton3.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.toolStripButton3.Image = ((System.Drawing.Image)(resources.GetObject("toolStripButton3.Image")));
            this.toolStripButton3.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.toolStripButton3.Name = "toolStripButton3";
            this.toolStripButton3.Size = new System.Drawing.Size(23, 22);
            this.toolStripButton3.Text = "Save Frame";
            this.toolStripButton3.Click += new System.EventHandler(this.toolStripButton3_Click);
            // 
            // tabControl1
            // 
            this.tabControl1.Controls.Add(this.tabPage1);
            this.tabControl1.Controls.Add(this.tabPage2);
            this.tabControl1.Controls.Add(this.tabPage3);
            this.tabControl1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tabControl1.Location = new System.Drawing.Point(0, 25);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(741, 635);
            this.tabControl1.TabIndex = 2;
            // 
            // tabPage1
            // 
            this.tabPage1.Controls.Add(this.RenderArea);
            this.tabPage1.Location = new System.Drawing.Point(4, 22);
            this.tabPage1.Name = "tabPage1";
            this.tabPage1.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage1.Size = new System.Drawing.Size(733, 609);
            this.tabPage1.TabIndex = 0;
            this.tabPage1.Text = "3D";
            this.tabPage1.UseVisualStyleBackColor = true;
            // 
            // tabPage2
            // 
            this.tabPage2.Controls.Add(this.canvas1);
            this.tabPage2.Location = new System.Drawing.Point(4, 22);
            this.tabPage2.Name = "tabPage2";
            this.tabPage2.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage2.Size = new System.Drawing.Size(733, 609);
            this.tabPage2.TabIndex = 1;
            this.tabPage2.Text = "2D";
            this.tabPage2.UseVisualStyleBackColor = true;
            // 
            // canvas1
            // 
            this.canvas1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.canvas1.Location = new System.Drawing.Point(3, 3);
            this.canvas1.Name = "canvas1";
            this.canvas1.Size = new System.Drawing.Size(727, 603);
            this.canvas1.TabIndex = 0;
            this.canvas1.Zoom = new System.Drawing.SizeF(1F, 1F);
            // 
            // tabPage3
            // 
            this.tabPage3.Controls.Add(this.canvas_ellipse);
            this.tabPage3.Controls.Add(this.toolStrip2);
            this.tabPage3.Location = new System.Drawing.Point(4, 22);
            this.tabPage3.Name = "tabPage3";
            this.tabPage3.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage3.Size = new System.Drawing.Size(733, 609);
            this.tabPage3.TabIndex = 2;
            this.tabPage3.Text = "EllipseExtraction";
            this.tabPage3.UseVisualStyleBackColor = true;
            // 
            // canvas_ellipse
            // 
            this.canvas_ellipse.Dock = System.Windows.Forms.DockStyle.Fill;
            this.canvas_ellipse.Location = new System.Drawing.Point(3, 34);
            this.canvas_ellipse.Name = "canvas_ellipse";
            this.canvas_ellipse.Size = new System.Drawing.Size(727, 572);
            this.canvas_ellipse.TabIndex = 1;
            this.canvas_ellipse.Zoom = new System.Drawing.SizeF(1F, 1F);
            this.canvas_ellipse.OnCanvasMouseClick += new FxMaths.GUI.Canvas.CanvasMouseClickHandler(this.canvas_ellipse_OnCanvasMouseClick);
            // 
            // toolStrip2
            // 
            this.toolStrip2.ImageScalingSize = new System.Drawing.Size(24, 24);
            this.toolStrip2.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripButton_EllipseOpenImage,
            this.toolStripButton_eclipse_SelectSubRegion,
            this.toolStripSeparator1,
            this.toolStripButton_ellipse_extract});
            this.toolStrip2.Location = new System.Drawing.Point(3, 3);
            this.toolStrip2.Name = "toolStrip2";
            this.toolStrip2.Size = new System.Drawing.Size(727, 31);
            this.toolStrip2.TabIndex = 0;
            this.toolStrip2.Text = "toolStrip2";
            // 
            // toolStripButton_EllipseOpenImage
            // 
            this.toolStripButton_EllipseOpenImage.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.toolStripButton_EllipseOpenImage.Image = ((System.Drawing.Image)(resources.GetObject("toolStripButton_EllipseOpenImage.Image")));
            this.toolStripButton_EllipseOpenImage.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.toolStripButton_EllipseOpenImage.Name = "toolStripButton_EllipseOpenImage";
            this.toolStripButton_EllipseOpenImage.Size = new System.Drawing.Size(28, 28);
            this.toolStripButton_EllipseOpenImage.Text = "Open Image";
            this.toolStripButton_EllipseOpenImage.Click += new System.EventHandler(this.toolStripButton_EllipseOpenImage_Click);
            // 
            // toolStripButton_eclipse_SelectSubRegion
            // 
            this.toolStripButton_eclipse_SelectSubRegion.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.toolStripButton_eclipse_SelectSubRegion.Image = global::Tester.Properties.Resources.select_rectangular;
            this.toolStripButton_eclipse_SelectSubRegion.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.toolStripButton_eclipse_SelectSubRegion.Name = "toolStripButton_eclipse_SelectSubRegion";
            this.toolStripButton_eclipse_SelectSubRegion.Size = new System.Drawing.Size(28, 28);
            this.toolStripButton_eclipse_SelectSubRegion.Text = "Select Sub Region";
            this.toolStripButton_eclipse_SelectSubRegion.ToolTipText = "Select Sub Region";
            this.toolStripButton_eclipse_SelectSubRegion.Click += new System.EventHandler(this.toolStripButton_eclipse_SelectSubRegion_Click);
            // 
            // toolStripSeparator1
            // 
            this.toolStripSeparator1.Name = "toolStripSeparator1";
            this.toolStripSeparator1.Size = new System.Drawing.Size(6, 31);
            // 
            // toolStripButton_ellipse_extract
            // 
            this.toolStripButton_ellipse_extract.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.toolStripButton_ellipse_extract.Image = ((System.Drawing.Image)(resources.GetObject("toolStripButton_ellipse_extract.Image")));
            this.toolStripButton_ellipse_extract.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.toolStripButton_ellipse_extract.Name = "toolStripButton_ellipse_extract";
            this.toolStripButton_ellipse_extract.Size = new System.Drawing.Size(28, 28);
            this.toolStripButton_ellipse_extract.Text = "Extract";
            this.toolStripButton_ellipse_extract.Click += new System.EventHandler(this.toolStripButton_ellipse_extract_Click);
            // 
            // KinectV2Form
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(741, 660);
            this.Controls.Add(this.tabControl1);
            this.Controls.Add(this.toolStrip1);
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(161)));
            this.Name = "KinectV2Form";
            this.Text = "KinectV2Form";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.KinectV2Form_FormClosing);
            ((System.ComponentModel.ISupportInitialize)(this.RenderArea)).EndInit();
            this.toolStrip1.ResumeLayout(false);
            this.toolStrip1.PerformLayout();
            this.tabControl1.ResumeLayout(false);
            this.tabPage1.ResumeLayout(false);
            this.tabPage2.ResumeLayout(false);
            this.tabPage3.ResumeLayout(false);
            this.tabPage3.PerformLayout();
            this.toolStrip2.ResumeLayout(false);
            this.toolStrip2.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.PictureBox RenderArea;
        private System.Windows.Forms.ToolStrip toolStrip1;
        private System.Windows.Forms.ToolStripButton toolStripButton1;
        private System.Windows.Forms.ToolStripLabel toolStripLabel1;
        private System.Windows.Forms.ToolStripLabel toolStripLabel_fps;
        private System.Windows.Forms.ToolStripButton toolStripButton2;
        private System.Windows.Forms.TabControl tabControl1;
        private System.Windows.Forms.TabPage tabPage1;
        private System.Windows.Forms.TabPage tabPage2;
        private System.Windows.Forms.ToolStripButton toolStripButton3;
        private FxMaths.GUI.Canvas canvas1;
        private System.Windows.Forms.TabPage tabPage3;
        private System.Windows.Forms.ToolStrip toolStrip2;
        private System.Windows.Forms.ToolStripButton toolStripButton_EllipseOpenImage;
        private FxMaths.GUI.Canvas canvas_ellipse;
        private System.Windows.Forms.ToolStripButton toolStripButton_eclipse_SelectSubRegion;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator1;
        private System.Windows.Forms.ToolStripButton toolStripButton_ellipse_extract;
    }
}