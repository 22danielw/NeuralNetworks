import java.io.*;
import java.util.Scanner;
import java.util.StringTokenizer;

/**
 * A class representing the A-B-C perceptron. Contains functionality to
 * configure the number of units in the activation and hidden layer as
 * well as the number of hidden layers, setting the initial weights of
 * the perceptron, and calculating the values of the final outputs. Furthermore,
 * the perceptron can train itself using multiple input values and their
 * corresponding expected values given by the user. Currently configured to work
 * only on an A-B-C perceptron.
 *
 * @author Daniel Wu
 * @version 03/05/2020
 * @version 03/23/2020
 * @version 04/14/2020
 */
public class Perceptron
{
   private final int NUM_LAYERS;        // the number of layers in the perceptron, given by user
   private double learningFactor;
   private double[][] units;            // the 2d array that represents the units of the perceptron
   private int[] layerSizes;            // stores the size of each layer in the perceptron
   private double[][][] weights;        // the 3d array that represents the weights of the perceptron
   private double[][][] deltaWeights;   // stores the values to change weights through each iteration, updates with each iteration
   private boolean debug = false;       // prints debug information if set to true
   private double[][] inputs;           // stores a number of inputs given by user's input and the size of the input layer
   private double[][] expectedValues;   // stores expected values given by user for all , each one corresponding to a set of inputs
   private double[][] finalValues;      // stores the final values of the all input vectors, updated during each calculation
   private int numInputVectors;         // the number of input vectors being tested per iteration
   private double[] currentErrorVals;   // stores the error values of each input vector within an iteration


   /**
    * Constructs a Perceptron based on parameters given by the user. Creates the structure of the perceptron
    * based on the number of layers and size of layers passed by the user, and initializes the weights, expected values,
    * delta weights, and final value arrays using a fully connected model. Also takes and stores a set learning factor from
    * the user and stores the input vectors used to run and train the network.
    *
    * The format of the structureConfig array is as follows:
    * [layerSize1, layerSize2, ... , layersizeN]
    *
    * @param numLayers       the number of layers in the perceptron
    * @param structureConfig an array containing the size of each layer in the perceptron
    * @param numInputVectors the number of input vectors that the perceptron will train with
    * @param learningFactor  the learning factor for the perceptron, decided by the user
    */
   public Perceptron(int numLayers, int[] structureConfig, int numInputVectors, double learningFactor)
   {
      NUM_LAYERS = numLayers;
      units = new double[NUM_LAYERS][]; // creates the perceptron with desired number of layers
      layerSizes = new int[NUM_LAYERS]; // creates the array storing layer sizes

      for (int n = 0; n < NUM_LAYERS; n++) // initializes the perceptron by creating layers of a given size and storing the sizes
      {
         units[n] = new double[structureConfig[n]];
         layerSizes[n] = structureConfig[n];
      }


      weights = new double[NUM_LAYERS - 1][][];

      for (int n = 0; n < weights.length; n++) // creates weights array based on sizes of layers
      {
         int currentLayerSize = layerSizes[n];
         int nextLayerSize = layerSizes[n + 1];
         weights[n] = new double[currentLayerSize][nextLayerSize];
      } // for (int n = 0; i < weights.length; n++)


      deltaWeights = new double[NUM_LAYERS - 1][][];

      for (int n = 0; n < deltaWeights.length; n++) // creates weights array based on sizes of layers
      {
         int currentLayerSize = layerSizes[n];
         int nextLayerSize = layerSizes[n + 1];
         deltaWeights[n] = new double[currentLayerSize][nextLayerSize];
      } // for (int n = 0; i < deltaWeights.length; n++)

      /*
       * The inputs array stores the inputs while the expectedValues stores the corresponding expected
       * value for any input vector with the same index i.
       */
      inputs = new double[numInputVectors][];

      for (int n = 0; n < numInputVectors; n++) // stores the input vectors
      {
         inputs[n] = new double[layerSizes[0]];
      }

      expectedValues = new double[numInputVectors][];

      for (int numInput = 0; numInput < numInputVectors; numInput++) // initializing the expected value storage array
      {
         expectedValues[numInput] = new double[layerSizes[NUM_LAYERS - 1]];
      }

      this.numInputVectors = numInputVectors;
      this.learningFactor = learningFactor;

      currentErrorVals = new double[numInputVectors];

      finalValues = new double[numInputVectors][];

      for (int inputNum = 0; inputNum < numInputVectors; inputNum++) // initializing the final value storage array
      {
         finalValues[inputNum] = new double[layerSizes[NUM_LAYERS - 1]];
      }

   } // public Perceptron(int numLayers, int[] structureConfig, int numInputVectors, double learningFactor)

   /**
    * Sets the inputs of network by taking the inputs given in the file and setting them
    * to the proper array (inputs array). Takes each set of inputs based on the size
    * of the initial layer and the total number of expected input sets.
    *
    * The precondition for this method is that the user knows the size of the perceptron and the number of expected input sets
    * gives inputs accordingly.
    *
    * The format of the input file is as follows where a is the initial inputs:
    *
    * a1 a2 ... (first input set)
    * a1 a2 ... (second input set)
    * ...etc
    *
    * @param inputFileName the name of the file containing the set of input vectors for the perceptron
    * @throws IOException if an IO error occurs
    */
   public void setInputs(String inputFileName) throws IOException
   {
      Scanner scanner = new Scanner(new File(inputFileName));

      for (int numInput = 0; numInput < numInputVectors; numInput++)
      {
         for (int k = 0; k < layerSizes[0]; k++) // parses user input, assigns vals to units[0][k]
         {
            inputs[numInput][k] = scanner.nextDouble();
         }
      }

   } // public void setInputs(String inputFileName) throws IOException

   /**
    * Sets the expected values of network by taking the values given in the file and setting them
    * to the proper array (expected vals array). Takes each set of expected values based on the size
    * of the final layer and the total number of expected input sets.
    *
    * The format of the input file is as follows where e is the expected value:
    *
    * e1 e2 ... (first input set)
    * e1 e2 ... (second input set)
    * ...etc
    *
    * @param expectedFileName the name of the file containing the sets of expected values of the perceptron
    * @throws IOException if an IO error occurs
    */
   public void setExpectedValues(String expectedFileName) throws IOException
   {
      Scanner scanner = new Scanner(new File(expectedFileName));

      for (int numInput = 0; numInput < numInputVectors; numInput++)
      {
         for (int i = 0; i < layerSizes[NUM_LAYERS - 1]; i++)
         {
            expectedValues[numInput][i] = scanner.nextDouble();
         }

      }

   } // public void setExpectedValues(String expectedFileName) throws IOException

   /**
    * Sets the weights for the perceptron. These weights are set sequentially into the array of
    * weights, where w[0][0][0] is the first input, w[0][0][1] is the second input, w[0][1][0] is the third, etc..
    *
    * A precondition of this method is that the user inputs the weights based on the size they
    * set. This method can be overloaded, and this version takes input from  a file.
    *
    * @param weightFile the file that is currently storing the weights for the perceptrons
    *
    * @throws IOException if an issue with IO is found
    */
   public void setWeights(String weightFile) throws IOException
   {
      Scanner scanner = new Scanner(new File(weightFile));

      for (int n = 0; n < NUM_LAYERS - 1; n++)
      {
         for (int k = 0; k < layerSizes[n]; k++)
         {
            for (int j = 0; j < layerSizes[n + 1]; j++)
            {
               weights[n][k][j] = scanner.nextDouble();
               if (debug)
                  System.out.println("DEBUG: w[" + n + "][" + k + "][" + j + "] = " + weights[n][k][j]);
            }
         }
      }

      if (debug)
         System.out.println("\n");

   } // public void setWeights(String weightFile) throws IOException

   /**
    * Sets the weights for the perceptron. These weights are set sequentially into the array of
    * weights, where w[0][0][0] is the first input, w[0][0][1] is the second input, w[0][1][0] is the third, etc..
    *
    * A precondition of this method is that the user inputs the weights based on the size they
    * set. This method can be overloaded, and this version takes an upper and lower limit and generates
    * random values for the weights.
    *
    * @param lowerLimit the lower limit of the random number generator
    * @param upperLimit the upper limit of the random number generator
    */
   public void setWeights(double lowerLimit, double upperLimit)
   {
      for (int n = 0; n < NUM_LAYERS - 1; n++)
      {
         for (int k = 0; k < layerSizes[n]; k++)
         {
            for (int j = 0; j < layerSizes[n + 1]; j++)
            {
               weights[n][k][j] = (Math.random() * (upperLimit - lowerLimit)) + lowerLimit;
            }
         }
      }
   } // public void setWeights(double lowerLimit, double upperLimit)

   /**
    * Calculates the value of the final unit for the current training set. This method updates unit using
    * the forwardUpdateUnit starting from the first hidden layer and ending with the
    * output layer. It then assigns the calculated value of the final unit for the current training set.
    *
    * @param inputSet the input set that is currently being run
    */
   public void calculateOutput(int inputSet)
   {
      for (int k = 0; k < layerSizes[0]; k++)
      {
         units[0][k] = inputs[inputSet][k];
      }

      for (int n = 1; n < NUM_LAYERS; n++) // updates units starting from the first hidden layer
      {
         for (int m = 0; m < layerSizes[n]; m++)
         {
            forwardUpdateUnit(n, m);
         }
      }

      for (int i = 0; i < layerSizes[NUM_LAYERS - 1]; i++)
      {
         finalValues[inputSet][i] = units[NUM_LAYERS - 1][i];
      }

   } // public void calculateOutput(int inputSet)

   /**
    * Calculates the error of the perceptron based on the equation given in the documentation. Performs a summation
    * over every (expected - final val)^2 in a given input set, then divides that by 2. The method then returns this
    * calculation.
    *
    * A precondition for this method is that calculateOutput() has already been called and the
    * the final units' values have already been calculated (and stored in the finalValues[] array).
    *
    * @param inputSet the input set over which the error is being calculated
    * @return the output of the error function of the perceptron for a given input set.
    */
   public double calculateError(int inputSet)
   {
      double rawError = 0;

      for (int i = 0; i < layerSizes[NUM_LAYERS - 1]; i++)
      {
         rawError += expectedValues[inputSet][i] - finalValues[inputSet][i];
      }
      return (rawError * rawError) / 2.0;
   } // public double calculateError(int inputSet)

   /**
    * Updates a unit at a given location in a given layer. It adds takes the weighted
    * sum of all units in the previous layer, then takes an activation function on
    * the result and maps it to the unit at the given location. Also creates a debug
    * message and prints it if the program is set to debugging mode.
    *
    * A precondition for this method is that it is not called on layer 0 (input layer).
    *
    * @param layer    the layer that the modified unit is in
    * @param location the location of the modified unit within its layer
    */
   private void forwardUpdateUnit(int layer, int location)
   {
      double netInput = 0.0;
      String debugMessage = "";

      for (int k = 0; k < layerSizes[layer - 1]; k++) // creates a net input
      {
         netInput += units[layer - 1][k] * weights[layer - 1][k][location];
         debugMessage += "a[" + (layer - 1) + "][" + k + "] * w[" + (layer - 1) +
               "][" + k + "][" + location + "] + ";
      }

      units[layer][location] = activationFunction(netInput); // sets output layer to f(net input)

      if (debug) // prints debug message that prints the relationships between units
      {
         debugMessage = debugMessage.substring(0, debugMessage.length() - 2); // Cleaning message
         System.out.println("DEBUG: a[" + layer + "][" + location + "] = " + debugMessage);
      }

   } // private void forwardUpdateUnit(int layer, int location)


   /**
    * Updates the weights in the perceptron by iterating through each weight and adding the corresponding
    * adjustment value stored in the deltaWeights[] array. Then, it writes each of the weights to a file
    * with a name given by a user, each one on a new line, to save them.
    *
    * @param inputNum the input set that is used to calculate the delta weights of the perceptron
    * @param outFile the file that this method outputs to
    * @throws IOException if an output error is detected
    */
   public void updateWeights(int inputNum, String outFile) throws IOException
   {
      calculateDeltaWeights(inputNum);

      PrintWriter writer = new PrintWriter(new File(outFile));

      for (int n = 0; n < NUM_LAYERS - 1; n++)
      {
         for (int k = 0; k < layerSizes[n]; k++)
         {
            for (int j = 0; j < layerSizes[n + 1]; j++)
            {
               weights[n][k][j] += deltaWeights[n][k][j];
               writer.write(weights[n][k][j] + "\n"); // writes the weights to file, each one on a new line
            }
         }
      }

      writer.flush();
      writer.close();
   } // public void updateWeights(int inputNum, String outFile) throws IOException

   /**
    * Calculates the adjustment values of each weight in the perceptron. This method does so
    * by calculating iterating through each weight in the hidden layer connected to each weight in the final layer
    * first, then using a separate loop construct to iterate through remaining layers of the perceptron.
    * It then stores each value into an array, with one adjustment value for each weight.
    * Currently configured to run on an A-B-C perceptron.
    *
    * @param inputNum the set of inputs over which the weight adjustment values are calculated.
    */
   public void calculateDeltaWeights(int inputNum)
   {
      for (int k = 0; k < layerSizes[NUM_LAYERS - 2]; k++) // iterates through the weights in the final layer
      {
         for (int i = 0; i < layerSizes[NUM_LAYERS - 1]; i++)
         {
            deltaWeights[NUM_LAYERS - 2][k][i] = getFinalAdjustmentValue(inputNum, k, i); // sets adjustment values of final layer
            if (debug)
               System.out.println("DEBUG: Delta[" + (NUM_LAYERS - 2) + "][" + k + "][" + i + "]: " +
                     deltaWeights[NUM_LAYERS - 2][k][i]);
         }
      } // for (int k = 0; k < layerSizes[NUM_LAYERS - 2]; k++)

      for (int n = NUM_LAYERS - 3; n >= 0; n--) // iterates through the hidden layers from right to left
      {
         for (int k = 0; k < layerSizes[n]; k++)
         {
            for (int j = 0; j < layerSizes[n + 1]; j++)
            {
               deltaWeights[n][k][j] = getHiddenAdjustmentValue(inputNum, n, k, j); // sets the adjustment value of hidden layers
               if (debug)
               {
                  System.out.println("DEBUG: Delta[" + n + "][" + k + "][" + j + "]: " +
                        deltaWeights[n][k][j]);
               }

            } // for (int j = 0; j < layerSizes[n + 1]; j++)

         } // for (int k = 0; k < layerSizes[n]; k++)

      } // for (int n = NUM_LAYERS - 3; n >= 0; n--)

   } // public void calculateDeltaWeights(int inputNum)

   /**
    * The method that returns the adjustment value for the weights connected to the final layer of the perceptron.
    * Returns the value for a given weight that is identified by the previous and next units it is connected to.
    * This method is currently set to work in an A-B-C network, so some code is specific to that configuration.
    * Calculates the value by multiplying the partial derivative of the error function with respect to the
    * current weight by the learning factor, as shown in the documentation. The variable names and constructs are
    * not optimized; they are configured to mirror the documentation provided exactly. This method returns the final
    * calculation (adjustment value).
    *
    * @param inputNum the input set over which the adjustment values for the final connectivity layer are calculated
    * @param currentIndex the position of the unit that the weight is attached to in the input layer
    * @param nextIndex the position of the unit that the weight is attached to in the output layer
    * @return the adjustment value for the weight based on the current activation state
    */
   private double getFinalAdjustmentValue(int inputNum, int currentIndex, int nextIndex)
   {
      double capitalThetai = 0.0;

      for (int k = 0; k < layerSizes[NUM_LAYERS - 2]; k++) // creates a net input based on previous layer
      {
         capitalThetai += units[NUM_LAYERS - 2][k] * weights[NUM_LAYERS - 2][k][nextIndex];
      }

      double partialDerivError = expectedValues[inputNum][nextIndex] - finalValues[inputNum][nextIndex];
      partialDerivError *= activationFunctionDerivative(capitalThetai);
      partialDerivError *= -units[NUM_LAYERS - 2][currentIndex];
      return -partialDerivError * learningFactor; // accumulator for the derivative eventually multiplied by learning factor
   } // private double getFinalAdjustmentValue(int inputNum, int currentIndex, int nextIndex)

   /**
    * The method that returns the adjustment value for the weight of any hidden layer of the perceptron.
    * Returns the value for a given weight that is identified by the previous and next units it is connected to
    * as well as the layer that the weight is in (layer number of the input unit)
    * Calculates the value by multiplying the partial derivative of the error function with respect to the
    * current weight by the learning factor, as shown in the documentation. The code is not optimized; it is implemented
    * to mirror the documentation provided exactly. This method returns the final calculation (adjustment value).
    *
    * @param inputNum the input set over which the adjustment values for the hidden layer are calculated
    * @param layerNum the connectivity layer that the weight is in
    * @param currentIndex the position of the unit that the weight is attached to in the input layer
    * @param nextIndex the position of the unit that the weight is attached to in the output layer
    * @return the adjustment value for the weight based on the current activation state
    */
   private double getHiddenAdjustmentValue(int inputNum, int layerNum, int currentIndex, int nextIndex)
   {
      double capitalThetaj = 0.0;

      for (int k = 0; k < layerSizes[layerNum]; k++) // creates a net input based on previous layer
      {
         capitalThetaj += units[layerNum][k] * weights[layerNum][k][nextIndex];
      }

      double capitalOmegaj = 0.0;

      for (int i = 0; i < layerSizes[NUM_LAYERS - 1]; i++)
      {
         double capitalThetai = 0.0;

         for (int k = 0; k < layerSizes[NUM_LAYERS - 2]; k++) // creates a net input based on previous layer
         {
            capitalThetai += units[NUM_LAYERS - 2][k] * weights[NUM_LAYERS - 2][k][i];
         }

         double lowercaseOmegai =  expectedValues[inputNum][i] - units[NUM_LAYERS - 1][i];
         capitalOmegaj += (activationFunctionDerivative(capitalThetai) * lowercaseOmegai)
               * weights[NUM_LAYERS - 2][nextIndex][i];
      } // for (int i = 0; i < layerSizes[NUM_LAYERS - 1]; i++)

      double partialDerivError = -units[layerNum][currentIndex];
      partialDerivError *= activationFunctionDerivative(capitalThetaj) * capitalOmegaj;
      return partialDerivError * -learningFactor; // accumulator of the derivative eventually multiplied by learning factor

   } // private double getHiddenAdjustmentValue(int inputNum, int layerNum, int currentIndex, int nextIndex)

   /**
    * Trains the perceptron by sequentially evaluating the perceptron for each input vector. After each
    * iteration, it updates the weights and outputs them to a given output file. One iteration consists of
    * calculating the output layer with one input/expected value set and, with the same set, updating the weights
    * of the perceptron. These new weight values are stored and used to run the next input vector.
    *
    *
    * @param outputFile the file to which weights are written to after each evaluation
    * @throws IOException if an issue with IO is found
    */
   public void train(String outputFile) throws IOException
   {
      for (int inputNum = 0; inputNum < numInputVectors; inputNum++)
      {
         calculateOutput(inputNum);
         currentErrorVals[inputNum] = calculateError(inputNum);
         updateWeights(inputNum, outputFile);
      } // for (int inputNum = 0; inputNum < numInputVectors; inputNum++)
   } // public void train(String outputFile) throws IOException

   /**
    * Prints the outputs of the perceptron for each input vector on a new line.
    */
   public void printOutputs()
   {
      for (int inputNum = 0; inputNum < numInputVectors; inputNum++)
      {
         for (int i = 0; i < layerSizes[NUM_LAYERS - 1]; i++)
         {
            System.out.println("Output " + i + " for input set " + inputNum + " is: " + finalValues[inputNum][i]
                               + " (should be " + expectedValues[inputNum][i] + ")");
         }
         System.out.print("\n");
      }
   }

   /**
    * Returns the "total error" of the perceptron, defined as the sum of the errors produced by
    * running each input vector.
    *
    * @return the total error of the perceptron
    */
   public double getTotalError()
   {
      double totalError = 0.0;

      for (int inputNum = 0; inputNum < numInputVectors; inputNum++)
      {
         totalError += currentErrorVals[inputNum];
      }

      return totalError;
   } // public double getTotalError()

   /**
    * Performs an activation function on a given net input. The activation function
    * is taken on a net input, and represents f(x) = 1 / (1 + e^(-x))
    * The derivative of this activation function is f'(x) = f(x) * (1 - f(x))
    *
    * @param input the input of the activation function
    * @return the output of the activation function
    */
   private double activationFunction(double input)
   {
      return 1.0 / (1.0 + Math.exp(-input)); // sigmoid function
   }

   /**
    * Performs the derivative of the activation function on a given input. The activation
    * function is: f(x) = 1 / (1 + e^(-x)).
    * The derivative of this activation function is: f'(x) = f(x) * (1 - f(x)).
    *
    * @param input the input of the derivative of the activation function
    * @return the derivative of the activation function at the point
    */
   private double activationFunctionDerivative(double input)
   {
      double activationOfInput = activationFunction(input);
      return activationOfInput * (1.0 - activationOfInput);
   }

   /**
    * Prints all the values of the weights in the Perceptron at a given time.
    * The format in which the weights are printed is as follows:
    * w[0][0][0] = value0
    * w[0][0][1] = value1
    * w[0][1][0] = value2
    * ... etc
    */
   public void printWeights()
   {
      for (int n = 0; n < weights.length; n++)
      {
         for (int k = 0; k < weights[n].length; k++)
         {
            for (int j = 0; j < weights[n][k].length; j++)
            {
               System.out.println("The value of w[" + n + "][" + k + "][" +
                     j + "] = " + weights[n][k][j]);
            }
         }
      }
   } // public void printWeights()

   /**
    * Prints all the values of the activations in the Perceptron at a given time.
    * The format in which the activations are printed is as follows:
    * a[0][0] = value0
    * a[0][1] = value1
    * a[1][0] = value2
    * ... etc
    */
   public void printActivations()
   {
      for (int n = 0; n < NUM_LAYERS; n++)
      {
         for (int k = 0; k < layerSizes[n]; k++)
         {
            System.out.println("The value of a[" + n + "][" + k + "] = " + units[n][k]);
         }
      }
   } // public void printActivations()

   /**
    * Returns the array containing the size of each layer in the perceptron. Accessor method.
    *
    * @return the layerSizes array
    */
   public int[] getLayerSizes()
   {
      return layerSizes;
   }

   /**
    * Returns the learning factor of the perceptron. Accessor method.
    *
    * @return the learningFactor instance variable
    */
   public double getLearningFactor()
   {
      return learningFactor;
   }
}
