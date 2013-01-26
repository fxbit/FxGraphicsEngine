namespace Tester
{
    partial class Viewport2D
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
            this.canvas1 = new FxMaths.GUI.Canvas();
            this.SuspendLayout();
            // 
            // canvas1
            // 
            this.canvas1.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.canvas1.EditBorderColor = new SharpDX.Color(((byte)(165)), ((byte)(42)), ((byte)(42)), ((byte)(255)));
            this.canvas1.Location = new System.Drawing.Point(-1, -1);
            this.canvas1.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.canvas1.Name = "canvas1";
            this.canvas1.SelectedBorderColor = new SharpDX.Color(((byte)(245)), ((byte)(245)), ((byte)(220)), ((byte)(255)));
            this.canvas1.Size = new System.Drawing.Size(285, 256);
            this.canvas1.TabIndex = 0;
            // 
            // Viewport2D
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(282, 253);
            this.Controls.Add(this.canvas1);
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 7.8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(161)));
            this.Name = "Viewport2D";
            this.Text = "Viewport2D";
            this.ResizeEnd += new System.EventHandler(this.Viewport2D_ResizeEnd);
            this.ResumeLayout(false);

        }

        #endregion

        private FxMaths.GUI.Canvas canvas1;
    }
}