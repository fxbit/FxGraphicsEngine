using FxMaths.Matrix;
using FxMaths.Vector;
using System;
using System.Collections.Generic;
using System.IO.Ports;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace SmartCam
{
    public class SerialCamera
    {
        /// <summary>
        /// Thread for reading from serial Port.
        /// </summary>
        Thread readThread;

        // Flag for the read thread reading.
        Boolean _continue = false;


        /// <summary>
        /// The serial port that communicate with that camera
        /// </summary>
        SerialPort serialPort;


        /// <summary>
        /// Save the recv data from the camera.
        /// </summary>
        FxMatrixMask imageMask;

        /// <summary>
        /// Event when we have recv new frame from camera.
        /// </summary>
        public event EventHandler<FxMatrixMask> HandleNewFrame;


        /// <summary>
        /// The center position of the camera.
        /// </summary>
        public FxVector2f Position { get; set; }

        /// <summary>
        /// The size of area tha the camera can see in the shop.
        /// </summary>
        public FxVector2f Size { get; set; }


        public SerialCamera(String portName,
            FxVector2f Position,
            FxVector2f Size)
        {

            // init the serial port
            serialPort = new SerialPort();

            // Set the read/write timeouts
            serialPort.ReadTimeout = -1;
            serialPort.WriteTimeout = -1;
            serialPort.Parity = Parity.None;
            serialPort.DataBits = 8;
            serialPort.StopBits = StopBits.Two;
            serialPort.Handshake = Handshake.None;
            serialPort.BaudRate = 921600;
            serialPort.NewLine = "\r\n";
            serialPort.PortName = portName;

            // Link the information of the camera to the place
            this.Position = Position;
            this.Size = Size;
        }



        /// <summary>
        /// Open the communication with camera.
        /// </summary>
        public Boolean Open()
        {
            try
            {
                serialPort.Open();
            }
            catch (Exception ex)
            {
                // debug messages
                Console.WriteLine(ex.StackTrace + ": " + ex.Message);
                _continue = false;
                return false;
            }
            try
            {
                _continue = true;

                if (readThread == null)
                    readThread = new Thread(Read);

                // check if the thread have be stoped
                if (readThread.ThreadState == System.Threading.ThreadState.Aborted || readThread.ThreadState == System.Threading.ThreadState.Stopped)
                {
                    readThread = new Thread(Read);
                    readThread.Priority = ThreadPriority.Highest;
                    readThread.Start();
                }
                else
                    readThread.Start();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.StackTrace + ": " + ex.Message);
                serialPort.Close();
                _continue = false;
                return false;
            }

            // flag the connection
            Console.WriteLine("Serial \"{0}\" Connected", serialPort.PortName);

            return true;
        }





        #region Read thread

        private void Read()
        {
            int count = 0;
            Byte[] buffer = new Byte[129];
            int numBytes = 10;
            Byte[,] imageBytes = new Byte[129, numBytes];

            int row_id = 0;

            while (_continue)
            {
                try
                {
                    // Read one row
                    row_id = readRow(buffer, numBytes) - 32;

                    // save the row
                    if (row_id >= 0 &&
                        row_id < 256)
                    {
                        for (int i = 0; i < numBytes; i++)
                            imageBytes[row_id, i] = buffer[i];
                    }

                    // Show results 
                    if (row_id == 63)
                    {
                        //Console.WriteLine("Read Image");

                        for (int i = 0; i < 64; i++)
                        {
                            int bindex = 0;
                            for (int j = 0; j < 64; j++)
                            {
                                byte b = imageBytes[i, bindex];

                                // Select the bit 
                                imageMask[j, i] = ((b & (1 << 7 - j % 8)) > 0);

                                // Move to the next byte
                                if (j % 8 == 7)
                                    bindex++;
                            }
                        }


                        // Now we have one full image we can send it for processing...
                        if (HandleNewFrame != null)
                            HandleNewFrame(this, imageMask);

                    }
                }
                catch (Exception ex) { Console.WriteLine(ex.Message); }
            }

            Console.WriteLine(count);
        }

        private int readRow(Byte[] buffer, int numBytes)
        {
            // Wait start chararcter
            char sc = '#';
            while (sc != '>')
                sc = (char)serialPort.ReadChar();

            //Console.WriteLine("Start Frame:");

            // Wait the line id
            int ind = serialPort.ReadByte();
            //Console.WriteLine("Received Row: " + ind.ToString());

            for (int i = 0; i < numBytes; i++)
                buffer[i] = (byte)serialPort.ReadByte();

            return ind;
        }

        #endregion
    }
}
