using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using FxMaths.GUI;
using FxMaths.Vector;
using FxMaths.Matrix;
using System.Threading;


namespace MainForm
{
    public class Person
    {
        #region Properties
        public FxVector2f Direction;
        public FxVector2f Position;
        public float Speed;
        #endregion



        #region Constructor
        public Person()
        {
            Position = new FxVector2f(); Direction = new FxVector2f(); Speed = 0.0f;
        }

        public Person(FxVector2f Position, FxVector2f Direction, float Speed)
        {
            this.Position = Position; this.Direction = Direction; this.Speed = Speed;
            this.Direction.Normalize();
        }
        #endregion


        public void Move(float speedMultiplier, float directionAngleChange)
        {
            Position += Speed * Direction;
            Speed *= speedMultiplier;
            Direction.Rotation(directionAngleChange);
        }

    }


    public class PeopleSimulation
    {
        private int _peopleNum = 0;
        public int PeopleNum { get { return _peopleNum; } }

        public FxVector2f EntrancePosition { get; set; }
        public FxVector2f EntranceDirection { get; set; }

        public List<Person> PersonList;


        private Boolean simulationRun = false;
        private Random rand;
        private Thread simulationThread = null;
        private FxMatrixF map = null;
        private FxMatrixMask mapMask = null;

        public delegate void PeopleSimulationRefresh(PeopleSimulation sim);
        private PeopleSimulationRefresh PsrList;

        /// <summary>
        /// Init simulation with specific number of runtime people.
        /// </summary>
        /// <param name="PeopleNum"></param>
        public PeopleSimulation(int PeopleNum, FxVector2f entrancePosition, FxVector2f entranceDirection, FxMatrixF im)
        {
            _peopleNum = PeopleNum;
            EntrancePosition = entrancePosition;
            EntranceDirection = entranceDirection;

            // init the blob list that contain the people
            PersonList = new List<Person>();

            // init random generator
            rand = new Random();

            // set the map mask
            map = im;
            mapMask = im > 0.8f;
        }

        private void SimulationRun()
        {
            while (simulationRun)
            {
                // if we can add people add them
                if (PersonList.Count < _peopleNum)
                {
                    int numOfNewPersons = rand.Next(5);
                    for (int i = 0; i < numOfNewPersons; i++)
                        PersonList.Add(new Person(EntrancePosition, EntranceDirection, 5*rand.Next(1000)/1000.0f + 2));
                }

                // move the blobs in random directions
                foreach (var person in PersonList)
                {
                    float speedChange = 1 + (rand.Next(100) > 50 ? -0.2f : +0.2f) * rand.Next(1000) / 1000.0f;
                    float directionAngleChange = (rand.Next(100) > 50 ? -1 : +1) * (float)(rand.NextDouble() * Math.PI/4.0f);

                    // move person
                    // check the person position and moving
                    FxVector2f nextPosition = person.Position + person.Speed * person.Direction;
                    float value = map[person.Position];
                    float valueNext = map[nextPosition];
                    
                    if(valueNext>0.9f)
                    {
                        person.Position = nextPosition;
                    }
                    else
                    {
                        directionAngleChange = (rand.Next(100) > 50 ? -1 : +1) * (float)(rand.NextDouble() * Math.PI);
                    }
                    person.Speed *= speedChange;
                    person.Direction.Rotation(directionAngleChange);
                }

                // call all upper layers about the refresh states
                PsrList(this);

                // delay the next frame
                Thread.Sleep(500);
            }
        }


        public void Start(PeopleSimulationRefresh psr)
        {
            if (simulationThread == null)
            {
                simulationThread = new Thread(new ThreadStart(SimulationRun));
                simulationRun = true;
                simulationThread.Start();
                PsrList += psr;
            }
        }

        public void Stop()
        {
            if(simulationThread!=null)
            {
                simulationRun = false;
                simulationThread.Abort();
                simulationThread.Join();
            }
        }
    }
}
