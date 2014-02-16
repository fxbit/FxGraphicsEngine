using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using FxMaths.GUI;
using FxMaths.Images;
using FxMaths.Geometry;
using FxMaths.Matrix;
using WeifenLuo.WinFormsUI.Docking;

namespace MainForm
{
    public partial class PeopleOverview : DockContent
    {
        GeometryPlotElement gpe;

        public PeopleOverview()
        {
            InitializeComponent();

            gpe = new GeometryPlotElement();
            canvas1.AddElements(gpe);

        }

        public PeopleOverview(FxMatrixF im)
        {
            InitializeComponent();

            // add building 
            ImageElement ie = new ImageElement(im, new ColorMap(ColorMapDefaults.Bones));
            canvas1.AddElements(ie);

            // add geometry plot
            gpe = new GeometryPlotElement();
            canvas1.AddElements(gpe);

        }

        private void PeopleOverview_Load(object sender, EventArgs e)
        {

        }

        public void PeopleUpdate(List<Person> personsList)
        {
            // remove all the old geometry
            gpe.ClearGeometry();

            // add all persons
            foreach (Person p in personsList)
            {
                Circle c = new Circle(p.Position, 10);
                gpe.AddGeometry(c, false);
            }

            gpe.ReDraw();
            
        }
    }
}
