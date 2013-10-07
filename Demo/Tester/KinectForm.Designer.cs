namespace Tester
{
    partial class KinectForm
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
            if(disposing && (components != null)) {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Component Designer generated code

        /// <summary> 
        /// Required method for Designer support - do not modify 
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(KinectForm));
            this.toolStrip1 = new System.Windows.Forms.ToolStrip();
            this.button_StartKinect = new System.Windows.Forms.ToolStripButton();
            this.canvas1 = new FxMaths.GUI.Canvas();
            this.statusStrip1 = new System.Windows.Forms.StatusStrip();
            this.toolStripStatusLabel1 = new System.Windows.Forms.ToolStripStatusLabel();
            this.labelFPS = new System.Windows.Forms.ToolStripStatusLabel();
            this.button_SaveImage = new System.Windows.Forms.ToolStripButton();
            this.toolStrip1.SuspendLayout();
            this.statusStrip1.SuspendLayout();
            this.SuspendLayout();
            // 
            // toolStrip1
            // 
            this.toolStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.button_StartKinect,
            this.button_SaveImage});
            this.toolStrip1.Location = new System.Drawing.Point(0, 0);
            this.toolStrip1.Name = "toolStrip1";
            this.toolStrip1.Size = new System.Drawing.Size(637, 25);
            this.toolStrip1.TabIndex = 0;
            this.toolStrip1.Text = "toolStrip1";
            // 
            // button_StartKinect
            // 
            this.button_StartKinect.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.button_StartKinect.Image = ((System.Drawing.Image)(resources.GetObject("button_StartKinect.Image")));
            this.button_StartKinect.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.button_StartKinect.Name = "button_StartKinect";
            this.button_StartKinect.Size = new System.Drawing.Size(23, 22);
            this.button_StartKinect.Text = "Start Kinect";
            this.button_StartKinect.Click += new System.EventHandler(this.button_StartKinect_Click);
            // 
            // canvas1
            // 
            this.canvas1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.canvas1.EditBorderColor = new SharpDX.Color(((byte)(165)), ((byte)(42)), ((byte)(42)), ((byte)(255)));
            this.canvas1.Location = new System.Drawing.Point(0, 25);
            this.canvas1.Name = "canvas1";
            this.canvas1.ScreenOffset = new SharpDX.Vector2(10F, 10F);
            this.canvas1.SelectedBorderColor = new SharpDX.Color(((byte)(245)), ((byte)(245)), ((byte)(220)), ((byte)(255)));
            this.canvas1.Size = new System.Drawing.Size(637, 431);
            this.canvas1.TabIndex = 1;
            this.canvas1.Zoom = new System.Drawing.SizeF(1F, 1F);
            // 
            // statusStrip1
            // 
            this.statusStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripStatusLabel1,
            this.labelFPS});
            this.statusStrip1.Location = new System.Drawing.Point(0, 434);
            this.statusStrip1.Name = "statusStrip1";
            this.statusStrip1.Size = new System.Drawing.Size(637, 22);
            this.statusStrip1.TabIndex = 2;
            this.statusStrip1.Text = "statusStrip1";
            // 
            // toolStripStatusLabel1
            // 
            this.toolStripStatusLabel1.Name = "toolStripStatusLabel1";
            this.toolStripStatusLabel1.Size = new System.Drawing.Size(29, 17);
            this.toolStripStatusLabel1.Text = "FPS:";
            // 
            // labelFPS
            // 
            this.labelFPS.Name = "labelFPS";
            this.labelFPS.Size = new System.Drawing.Size(13, 17);
            this.labelFPS.Text = "0";
            // 
            // button_SaveImage
            // 
            this.button_SaveImage.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.button_SaveImage.Image = ((System.Drawing.Image)(resources.GetObject("button_SaveImage.Image")));
            this.button_SaveImage.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.button_SaveImage.Name = "button_SaveImage";
            this.button_SaveImage.Size = new System.Drawing.Size(23, 22);
            this.button_SaveImage.Text = "SaveImage";
            this.button_SaveImage.Click += new System.EventHandler(this.button_SaveImage_Click);
            // 
            // KinectForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(637, 456);
            this.Controls.Add(this.statusStrip1);
            this.Controls.Add(this.canvas1);
            this.Controls.Add(this.toolStrip1);
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(161)));
            this.Name = "KinectForm";
            this.Text = "Kinect";
            this.Load += new System.EventHandler(this.KinectForm_Load);
            this.toolStrip1.ResumeLayout(false);
            this.toolStrip1.PerformLayout();
            this.statusStrip1.ResumeLayout(false);
            this.statusStrip1.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.ToolStrip toolStrip1;
        private System.Windows.Forms.ToolStripButton button_StartKinect;
        private FxMaths.GUI.Canvas canvas1;
        private System.Windows.Forms.StatusStrip statusStrip1;
        private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel1;
        private System.Windows.Forms.ToolStripStatusLabel labelFPS;
        private System.Windows.Forms.ToolStripButton button_SaveImage;
    }
}
