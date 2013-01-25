namespace SimpleExample
{
    partial class Form1
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
            this.start3d = new System.Windows.Forms.Button();
            this.RenderArea = new System.Windows.Forms.PictureBox();
            this.label_triangles = new System.Windows.Forms.Label();
            this.label_fps = new System.Windows.Forms.Label();
            this.timer1 = new System.Windows.Forms.Timer(this.components);
            this.button_create_mesh = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.RenderArea)).BeginInit();
            this.SuspendLayout();
            // 
            // start3d
            // 
            this.start3d.Location = new System.Drawing.Point(12, 12);
            this.start3d.Name = "start3d";
            this.start3d.Size = new System.Drawing.Size(75, 23);
            this.start3d.TabIndex = 0;
            this.start3d.Text = "Start 3D";
            this.start3d.UseVisualStyleBackColor = true;
            this.start3d.Click += new System.EventHandler(this.start3d_Click);
            // 
            // RenderArea
            // 
            this.RenderArea.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.RenderArea.Location = new System.Drawing.Point(93, 12);
            this.RenderArea.Name = "RenderArea";
            this.RenderArea.Size = new System.Drawing.Size(894, 488);
            this.RenderArea.TabIndex = 1;
            this.RenderArea.TabStop = false;
            this.RenderArea.MouseClick += new System.Windows.Forms.MouseEventHandler(this.RenderArea_MouseClick);
            this.RenderArea.Resize += new System.EventHandler(this.RenderArea_Resize);
            // 
            // label_triangles
            // 
            this.label_triangles.AutoSize = true;
            this.label_triangles.Location = new System.Drawing.Point(12, 467);
            this.label_triangles.Name = "label_triangles";
            this.label_triangles.Size = new System.Drawing.Size(59, 13);
            this.label_triangles.TabIndex = 11;
            this.label_triangles.Text = "Triangles:0";
            // 
            // label_fps
            // 
            this.label_fps.AutoSize = true;
            this.label_fps.Location = new System.Drawing.Point(12, 480);
            this.label_fps.Name = "label_fps";
            this.label_fps.Size = new System.Drawing.Size(39, 13);
            this.label_fps.TabIndex = 10;
            this.label_fps.Text = "FPS: 0";
            // 
            // timer1
            // 
            this.timer1.Enabled = true;
            this.timer1.Interval = 1000;
            this.timer1.Tick += new System.EventHandler(this.timer1_Tick);
            // 
            // button_create_mesh
            // 
            this.button_create_mesh.Location = new System.Drawing.Point(12, 53);
            this.button_create_mesh.Name = "button_create_mesh";
            this.button_create_mesh.Size = new System.Drawing.Size(75, 23);
            this.button_create_mesh.TabIndex = 12;
            this.button_create_mesh.Text = "Create Mesh";
            this.button_create_mesh.UseVisualStyleBackColor = true;
            this.button_create_mesh.Click += new System.EventHandler(this.button_create_mesh_Click);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(999, 512);
            this.Controls.Add(this.button_create_mesh);
            this.Controls.Add(this.label_triangles);
            this.Controls.Add(this.label_fps);
            this.Controls.Add(this.RenderArea);
            this.Controls.Add(this.start3d);
            this.Name = "Form1";
            this.Text = "Form1";
            ((System.ComponentModel.ISupportInitialize)(this.RenderArea)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button start3d;
        private System.Windows.Forms.PictureBox RenderArea;
        private System.Windows.Forms.Label label_triangles;
        private System.Windows.Forms.Label label_fps;
        private System.Windows.Forms.Timer timer1;
        private System.Windows.Forms.Button button_create_mesh;
    }
}

