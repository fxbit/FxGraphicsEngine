namespace Tester
{
    partial class Viewport
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
            this.RenderArea = new System.Windows.Forms.PictureBox();
            ((System.ComponentModel.ISupportInitialize)(this.RenderArea)).BeginInit();
            this.SuspendLayout();
            // 
            // RenderArea
            // 
            this.RenderArea.Dock = System.Windows.Forms.DockStyle.Fill;
            this.RenderArea.Location = new System.Drawing.Point(0, 0);
            this.RenderArea.Name = "RenderArea";
            this.RenderArea.Size = new System.Drawing.Size(783, 679);
            this.RenderArea.TabIndex = 0;
            this.RenderArea.TabStop = false;
            this.RenderArea.MouseClick += new System.Windows.Forms.MouseEventHandler(this.RenderArea_MouseClick);
            this.RenderArea.Resize += new System.EventHandler(this.RenderArea_Resize);
            // 
            // Viewport
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(783, 679);
            this.Controls.Add(this.RenderArea);
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 7.8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(161)));
            this.Name = "Viewport";
            this.Text = "Viewport";
            ((System.ComponentModel.ISupportInitialize)(this.RenderArea)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.PictureBox RenderArea;
    }
}