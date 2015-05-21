using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using FxMaths;
using FxMaths.Matrix;
using FxMaths.Vector;
using FxMaths.GUI;

using Accord.IO;
using AForge.Video;
using AForge.Video.FFMPEG;

using SmartCam;
using MoreLinq;
using System.Diagnostics;

namespace SmartCameraSimulation
{
    public partial class Form1 : Form
    {
        FxVector2i origRes = new FxVector2i(1920, 1080);
        FxVector2f resize;
        FxVector2i targetRes = new FxVector2i(720,576);

        FxMatrixF mat;
        ImageElement im;

        Shop shop;
        SimulationSimple simulation;

        FxMatrixF shopPlan;
        FxMatrixF imPerson;
        FxMatrixMask imPersonMask;
        Random rand = new Random();


        public Form1()
        {
            InitializeComponent();

            resize = new FxVector2f(targetRes.x / (float)origRes.x, targetRes.y / (float)origRes.y);

            // create a shop
            shopPlan = FxMatrixF.Load("Katopsi.jpg", ColorSpace.Grayscale);
            shopPlan = shopPlan.Resize(targetRes.x, targetRes.y);
            shop = new Shop("Unisol", shopPlan);

           // simulation = new SimulationSimple(shop, 10, simulationMethod1);

            simulationMethod2_Setup();
            simulation = new SimulationSimple(shop, 1, simulationMethod2);
            
            // Defind entrance for the shop
            shop.entrancePositionsDirection.Add(new Tuple<FxVector2f, FxVector2f>(new FxVector2f(1800 * resize.x, 500 * resize.y), new FxVector2f(-1, 0)));
            

            // load a person image for the moving image
            imPerson = FxMatrixF.Load("person.jpg", ColorSpace.Grayscale);
           // imPerson.Exec(x => (x > 0.1) ? 0 : 1);
            imPerson.MedianFilt();
            imPersonMask = imPerson < 0.5f;

            // Init the matrix to be HD
            mat = new FxMatrixF(targetRes.x, targetRes.y);
            mat.DrawMatrix(shopPlan, new FxVector2f(0, 0), new FxVector2f(mat.Width, mat.Height), FxMatrixF.DrawInterpolationMethod.NearestNeighbor);

            // Add image element to canvas to showing the matrix
            im = new ImageElement(mat);
            canvas1.AddElement(im, true);
        }





        #region Simulation Method 1 Random path
        private void simulationMethod1(List<Person> personList)
        {
            lock (shop)
            {
                FxVector2f filtered = new FxVector2f();
                // move the blobs in random directions
                foreach (var person in personList)
                {
                    float speedChange = 1 + (rand.Next(100) > 50 ? -0.1f : +0.1f) * rand.Next(1000) / 1000.0f;

                    float directionAngleChange = (rand.Next(100) >= 90 ? 0 : 1) *               // Select randomly if we are going to change the angle
                                                 (rand.Next(100) >= 50 ? -1 : +1) *             // Select randomly the left right.
                                                 (float)(rand.NextDouble() * Math.PI / 8.0f); // select the angle

                    // move person
                    // check the person position and moving
                    FxVector2f nextPosition = person.Position + person.Speed * person.Direction;
                    float value = shopPlan[person.Position];

                    if (nextPosition.x > 0 && nextPosition.y > 0 && nextPosition.x < shopPlan.Width && nextPosition.y < shopPlan.Height)
                    {
                        float valueNext = shopPlan[nextPosition];
                        if (valueNext > 0.9f)
                        {
                            // calculate next position
                            person.Path.Add(nextPosition);


                            // calculate the position with kalman to pe more smooth the transaction
                            filtered.x = person.kalmanX.Update(nextPosition.x, 50);
                            filtered.y = person.kalmanY.Update(nextPosition.y, 50);


                            person.Position = filtered;
                            person.PathKalman.Add(filtered);
                        }
                        else
                            directionAngleChange = (rand.Next(100) > 50 ? -1 : +1) * (float)(rand.NextDouble() * Math.PI);
                    }
                    else
                        directionAngleChange = (rand.Next(100) > 50 ? -1 : +1) * (float)(rand.NextDouble() * Math.PI);

                    // update the speed
                    person.Speed *= speedChange;

                    // limit the max speed
                    if (person.Speed > 6f)
                        person.Speed = 6;

                    // rotate the direction
                    person.Direction.Rotation(directionAngleChange);
                }
            }
        } 
        #endregion




        #region Simulation Method 1 Target selection

        List<FxVector2f> targetList;

        private void simulationMethod2_Setup()
        {
            targetList = new List<FxVector2f>();
            targetList.Add(new FxVector2f(900, 600));
            targetList.Add(new FxVector2f(1223, 158));
            targetList.Add(new FxVector2f(1415, 711));
            targetList.Add(new FxVector2f(1088, 676));
            targetList.Add(new FxVector2f(1095, 913));
            targetList.Add(new FxVector2f(1470, 865));
            targetList.Add(new FxVector2f(1676, 482));
            targetList.Add(new FxVector2f(1408.173f, 210.4465f));
            targetList.Add(new FxVector2f(1010.663f, 298.7821f));
            targetList.Add(new FxVector2f(1010.663f, 402.7064f));
            targetList.Add(new FxVector2f(974.289f, 545.6022f));
            targetList.Add(new FxVector2f(1228.904f, 423.4912f));
            targetList.Add(new FxVector2f(1564.060f, 342.9499f));
            targetList.Add(new FxVector2f(945.710f, 854.7767f));
            targetList.Add(new FxVector2f(810.608f, 693.6942f));
            targetList.Add(new FxVector2f(1706.955f, 922.3275f));

            // resize it
            for (int i = 0; i < targetList.Count; i++)
            {
                var t = targetList[i];
                t.x *= resize.x;
                t.y *= resize.y;
                targetList[i]= t;
            }
        }

        public void simulationMethod2(List<Person> personList)
        {

            lock (shop)
            {
                FxVector2f filtered = new FxVector2f();

                // move the blobs in smallest distance target
                foreach (var person in personList)
                {
                    // move person
                    // check the person position and moving
                    FxVector2f nextPosition = person.Position + person.Speed * person.Direction;
                    float value = shopPlan[person.Position];

                    float speedChange =  (rand.Next(100) > 50 ? -0.2f : +0.2f) * rand.Next(1000) / 1000.0f;


                    // select random target in the start
                    if (person.Target.x == 0 && person.Target.y == 0)
                    {
                        person.Target = targetList.RandomSelectStruct();
                    }

                    // check if the person arave to select target 
                    if (person.Target.Distance(ref person.Position) < 10)
                    {
                        if (person.waitInTarget &&
                            person.waitTime.ElapsedMilliseconds > person.waitTimeMs)
                        {
                            person.waitTime.Stop();

                            // select random target
                            person.Target = targetList.RandomSelectStruct();

                        }
                        else if (!person.waitInTarget)
                        {
                            person.waitInTarget = true;

                            // select a new random wait time 0-2Sec
                            person.waitTimeMs =  (long)(rand.NextDouble() * 2000);

                            person.waitTime = new Stopwatch();
                            person.waitTime.Start();
                        }
                        else
                        {
                            continue;
                        }
                    }
                    


                    // We are 1% "lucky" change the target
                    if(rand.Next(100) >= 100)
                    {
                        // select random target
                        person.Target = targetList.RandomSelectStruct();
                    }

                    // select the direction that you can go to the target
                    var targetDirection = person.Target - nextPosition;
                    float directionAngleChange = (rand.Next(100) >= 99 ? 0 : 1) *
                                                 (-(float)person.Direction.Angle(ref targetDirection));
                    /*
                                                 + 
                                                 (rand.Next(100) >= 50 ? -1 : +1)*              // Select randomly the left right.
                                                 (float)(rand.NextDouble() * Math.PI *(20f/360f)); // select the angle
                   */
                    directionAngleChange *= (float)rand.NextDouble()/2;

                    if (Math.Abs(directionAngleChange)>Math.PI/2)
                    {
                        Console.WriteLine("------------------------->");
                        /*
                        if (directionAngleChange < 0)
                            directionAngleChange = (float)(- directionAngleChange - Math.PI / 2);
                        else
                            directionAngleChange = (float)(- directionAngleChange + Math.PI / 2);
                         * */

                        directionAngleChange = -directionAngleChange;
                    }

                    if (directionAngleChange!=0)
                    {
                        Console.WriteLine(directionAngleChange);
                    }

                    if (float.IsNaN(directionAngleChange))
                        directionAngleChange = 0;
                    
                    if (nextPosition.x > 0 && nextPosition.y > 0 && nextPosition.x < shopPlan.Width && nextPosition.y < shopPlan.Height)
                    {
                        float valueNext = shopPlan[nextPosition];
                        if (valueNext > 0.9f)
                        {
                            // calculate next position
                            person.Path.Add(nextPosition);


                            // calculate the position with kalman to pe more smooth the transaction
                            filtered.x = person.kalmanX.Update(nextPosition.x, 50);
                            filtered.y = person.kalmanY.Update(nextPosition.y, 50);


                            person.Position = nextPosition;
                            person.PathKalman.Add(filtered);
                        }
                        else
                        {
                            directionAngleChange = (rand.Next(100) > 50 ? -1 : +1) * (float)(Math.PI);
                            // select random target
                            person.Target = targetList.RandomSelectStruct();
                        }
                    }
                    else
                        directionAngleChange = (rand.Next(100) > 50 ? -1 : +1) * (float)(Math.PI);

                    // update the speed
                    person.Speed += speedChange;

                    // limit the max speed
                    if (person.Speed > 8f)
                        person.Speed = 8;

                    // limit the min speed
                    if (person.Speed < 1f)
                        person.Speed = 1;

                    // rotate the direction
                    person.Direction.Rotation(directionAngleChange);
                }
            }

        }
        

        #endregion


        #region Simulation running
        private void toolStripButton_runSimulation_Click(object sender, EventArgs e)
        {
            // Execute async the video creation
            Task.Run((Action)CreateVideo);
        }



        private void CreateVideo()
        {
            // Start a new video file 
            VideoFileWriter vfw = new VideoFileWriter();
            vfw.Open("Test.avi", mat.Width, mat.Height, 30, VideoCodec.Default);

            // create an image that we are going to use for the buffering of the next frame
            Bitmap bitmap = new Bitmap(mat.Width, mat.Height);
            FxMaths.Images.FxImages nextFrameImage = FxMaths.Images.FxTools.FxImages_safe_constructors(bitmap);

            // Start the simulation
            simulation.Start();

            var imTarget = imPerson * 0.1f + 0.5f;

            for (int i = 0; i < 2000; i++)
            {
                // Set to zero value
                mat.DrawMatrix(shopPlan, new FxVector2f(0, 0), new FxVector2f(mat.Width, mat.Height), FxMatrixF.DrawInterpolationMethod.Linear);
                //mat.SetValue(1); // white


                // Draw the target points
                if(true)
                    foreach(var t in targetList)
                    {
                        var size = new FxVector2f(20 * resize.x, 20 * resize.y);
                        var center = size / 2;

                        // Draw a circle per person
                        mat.DrawMatrix(imTarget,
                            imPersonMask,
                            t - center,
                            size,
                            FxMatrixF.DrawInterpolationMethod.Linear);
                        
                    }



                lock (shop)
                {
                    var size = new FxVector2f(50 * resize.x, 50 * resize.y);
                    var center = size / 2;
                    // move the blobs in random directions
                    foreach (var person in shop.personList)
                    {
                        // Draw a circle per person
                        mat.DrawMatrix(imPerson,
                            imPersonMask,
                            person.Position - center,
                            size,
                            FxMatrixF.DrawInterpolationMethod.Linear);
                    }
                }

                // Update the bitmap that we write to image
                nextFrameImage.Load(mat, new FxMaths.Images.ColorMap(FxMaths.Images.ColorMapDefaults.Gray));
                vfw.WriteVideoFrame(bitmap);



                // Update showing image
                im.UpdateInternalImage(mat, new FxMaths.Images.ColorMap(FxMaths.Images.ColorMapDefaults.Gray));
                canvas1.ReDraw();

                if (i % 10 == 0)
                    Console.WriteLine("Write Frame:" + i.ToString());
            }

            // Stop the simulation
            simulation.Stop();
            vfw.Close();
        } 
        #endregion
    }
}
