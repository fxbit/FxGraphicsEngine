using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace GraphicsEngine.UI
{
    public partial class UISettings : Form
    {
        public UISettings()
        {
            InitializeComponent();
        }

        private void UISettings_Load( object sender, EventArgs e )
        {
            propertyGrid1.SelectedObject = new SettingsDummy();
        }

        private void button1_Click( object sender, EventArgs e )
        {
            this.Close();
        }

        private void button2_Click( object sender, EventArgs e )
        {
            this.Close();
        }
    }
}
