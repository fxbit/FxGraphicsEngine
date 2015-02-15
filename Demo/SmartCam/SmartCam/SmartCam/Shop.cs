using FxMaths.Matrix;
using FxMaths.Vector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SmartCam
{
    public class Shop
    {
        /// <summary>
        /// List with all cameras in the shop.
        /// </summary>
        List<SerialCamera> listCamera;


        /// <summary>
        /// The name of the shop.
        /// </summary>
        public String Name { get; set; }


        /// <summary>
        /// The Plan of the shop.
        /// </summary>
        public FxMatrixF ShopPlan;

        /// <summary>
        /// The state of the shop base on the camera input.
        /// </summary>
        public FxMatrixF ShopRuntimeState;


        /// <summary>
        /// The set with the active persons 
        /// that the shop have.
        /// </summary>
        public List<Person> personList;


        /// <summary>
        /// List with entrance that the shop have.
        /// </summary>
        public List<Tuple<FxVector2f, FxVector2f>> entrancePositionsDirection;

        public Shop(String Name, 
            FxMatrixF shopPlan)
        {
            this.Name = Name;
            listCamera = new List<SerialCamera>();
            personList = new List<Person>();
            entrancePositionsDirection = new List<Tuple<FxVector2f, FxVector2f>>();
            this.ShopPlan = shopPlan;

            // create a runtime state base on the size of the shop plan
            ShopRuntimeState = new FxMatrixF(shopPlan.Width, shopPlan.Height, 0);
        }



        public void AddCamera(SerialCamera camera)
        {
            // Add the camera to the list
            listCamera.Add(camera);

            // Get the event of the camera for processing
            camera.HandleNewFrame += camera_HandleNewFrame;
        }


        void camera_HandleNewFrame(object sender, FxMaths.Matrix.FxMatrixMask e)
        {
            SerialCamera cam = sender as SerialCamera;

            // lock base on shop runtime state because we can get parallel images
            lock (ShopRuntimeState)
            {
                FxVector2f start = cam.Position - cam.Size/2;

                // draw the image to bigger matrix
                ShopRuntimeState.DrawMatrix(e.ToFxMatrixF(), 
                    start, 
                    cam.Size,
                    FxMatrixF.DrawInterpolationMethod.NearestNeighbor);

                // TODO: process the image for extraction of persons.

            }
        }


    }
}
