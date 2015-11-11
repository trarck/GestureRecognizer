using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using NeuronDotNet.Core;
using NeuronDotNet.Core.Backpropagation;
using NeuronDotNet.Core.Initializers;

namespace GestureRecognizer
{
    public partial class MainForm : Form
    {

        private double learningRate = 0.3d;
        private int neuronCount = 10;
        private int cycles = 10000;
        private int inputVectorSize = 24;
        private int outVectorSize = 4;

        private BackpropagationNetwork network;

        public MainForm()
        {
            InitializeComponent();
        }

        void InitNeuralNet()
        {
            LinearLayer inputLayer = new LinearLayer(1);
            SigmoidLayer hiddenLayer = new SigmoidLayer(neuronCount);
            SigmoidLayer outputLayer = new SigmoidLayer(1);
            new BackpropagationConnector(inputLayer, hiddenLayer).Initializer = new RandomFunction(0d, 0.3d);
            new BackpropagationConnector(hiddenLayer, outputLayer).Initializer = new RandomFunction(0d, 0.3d);
            network = new BackpropagationNetwork(inputLayer, outputLayer);
            network.SetLearningRate(learningRate);

            TrainingSet trainingSet = new TrainingSet(inputVectorSize, outVectorSize);
            for (int i = 0; i < curve.Points.Count; i++)
            {
                double xVal = curve.Points[i].X;
                for (double input = xVal - 0.05; input < xVal + 0.06; input += 0.01)
                {
                    trainingSet.Add(new TrainingSample(new double[] { input }, new double[] { curve.Points[i].Y }));
                }
            }

            network.EndEpochEvent += new TrainingEpochEventHandler(
                delegate(object senderNetwork, TrainingEpochEventArgs args)
                {
                    trainingProgressBar.Value = (int)(args.TrainingIteration * 100d / cycles);
                    Application.DoEvents();
                });
            network.Learn(trainingSet, cycles);
        }

        private void button1_Click(object sender, EventArgs e)
        {

        }
    }
}
