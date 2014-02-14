using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using FxMaths.GUI;
using FxMaths.Vector;
using FxMaths.Matrix;


namespace MainForm
{
    public class PeopleSimulation
    {
        private int _peopleNum = 0;
        public int PeopleNum { get { return _peopleNum; } }


        public List<FxBlob> BlobList;
        private Boolean simulationRun = false;

        /// <summary>
        /// Init simulation with specific number of runtime people.
        /// </summary>
        /// <param name="PeopleNum"></param>
        public PeopleSimulation(int PeopleNum, FxVector2f entrancePosition, FxVector2f entranceDirection)
        {
            _peopleNum = PeopleNum;
            BlobList = new List<FxBlob>();
        }

        private void SimulationRun()
        {
            while (simulationRun)
            {

                System.Threading.Thread.Sleep(100);
            }
        }



    }
}
