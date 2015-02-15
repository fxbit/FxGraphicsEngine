
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Diagnostics;

using FxMaths;
using FxMaths.Vector;

namespace SmartCam
{
    public class SimulationSimple
    {
        private Shop shop = null;
        private Boolean _threadRunning = false;
        private Thread simulationThread = null;
        private int _numPeople = 0;
        private Action<List<Person>> peopleSimulation;


        /// <summary>
        /// Return the number of people 
        /// that are active to the shop.
        /// </summary>
        public int NumPeople { get { return this.shop.personList.Count; } }


        #region Constarctor
        public SimulationSimple(Shop shop, int numPeople, Action<List<Person>> peopleSimulation)
        {
            this.shop = shop;
            this._numPeople = numPeople;
            this.peopleSimulation = peopleSimulation;
        }
        
        #endregion





        #region Simulation Start/Stop
        public void Start()
        {           
            if (simulationThread == null)
            {
                // start the thread
                simulationThread = new Thread(new ThreadStart(SimulationThread));
                _threadRunning = true;
                simulationThread.Start();
            }
        }


        public void Stop()
        {
            if (simulationThread != null)
            {
                _threadRunning = false;
                simulationThread.Abort();
                simulationThread.Join();
            }

        } 
        #endregion





        #region Simulation Thread running

        private void SimulationThread()
        {
            Stopwatch addTime = new Stopwatch();
            addTime.Start();
            long waitTimeMs = 100;
            Random rand = new Random();
            int dt_ms = 50;

            while (_threadRunning)
            {
                lock (shop)
                {
                    // if we need more people add one now and
                    // wait rundom time for the next one.
                    if ((_numPeople > shop.personList.Count) &&
                        (addTime.ElapsedMilliseconds > waitTimeMs))
                    {
                        // select a new random wait time 0-1Sec
                        waitTimeMs = (long)(rand.NextDouble() * 1000);

                        // start again for the next add
                        addTime.Restart();

                        // Select the entrance to for the new person.
                        var entrance = shop.entrancePositionsDirection.RandomSelect();
                        if (entrance != null)
                        {
                            // add a new person to shop
                            shop.personList.Add(new Person(entrance.Item1, entrance.Item2, 4.0f * rand.Next(1000) / 1000.0f + 2.0f));
                        }
                    }
                }


                // Run the simulation
                if(peopleSimulation!=null)
                    peopleSimulation(shop.personList);


                // delay the next frame
                Thread.Sleep(dt_ms);
            }
        }


        #endregion
    }
}
