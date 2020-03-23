import java.io.*;
import java.util.Scanner;
import java.util.StringTokenizer;

/**
 * A class representing the A-B-1 perceptron. Contains functionality to
 * configure the number of units in the activation and hidden layer as
 * well as the number of hidden layers, setting the initial weights of
 * the perceptron, and calculating the value of the final output. Furthermore,
 * the perceptron can train itself using multiple input values and their
 * corresponding expected values. Currently configured to work only on an
 * A-B-1 perceptron.
 *
 * @author Daniel Wu
 * @version 03/05/2020
 * @version 03/23/2020
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
   private double[] expectedValues;     // stores expected values given by user, each one corresponding to a set of inputs
   private double currentExpectedValue; // stores the expected value of the current input vector, updated for each input vectorx
   private double currentFinalValue;    // stores the final value of the current input vector, updated for each input vector
   private int numInputVectors;         // the number of input vectors being tested per iteration
   private double[] currentErrorVals;   // stores the error values of each input vector within an iteration


   /**
    * Constructs a Perceptron based on parameters given by the user. Creates the structure of the perceptron
    * based on the number of layers and size of layers passed by the user, and initializes the weights array using
    * a fully connected model. Also takes and stores a set learning factor from the user and stores the input
    * vectors used to run and train the network.
    *
    * The format of the structureConfig array is as follows:
    * [layerSize1, layerSize2, ... , layersizeN]
    *
    * @param numLayers the number of layers in the perceptron
    * @param structureConfig an array containing the size of each layer in the perceptron
    * @param numInputVectors the number of input vectors that the perceptron will train with
    * @param learningFactor the learning factor for the perceptron, decided by the user
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
      expectedValues = new double[numInputVectors];
      this.numInputVectors = numInputVectors;

      this.learningFactor = learningFactor;
      currentErrorVals = new double[numInputVectors];
   } // public Perceptron(int numLayers, int[] structureConfig, int numInputVectors, double learningFactor)

   /**
    * The format of the config file is as follows where n is the number of input vectors,
    * a is the initial inputs, and e is the expected values. Sets the inputs of network by taking
    * the inputs and their expected output sequentially.
    *
    * The precondition for this method is that the user knows the size of the perceptron and
    * gives inputs accordingly.
    *
    * a01 a02 e1
    * a11 a12 e2
    * ...etc
    */
   public void setInputs(String configFile) throws IOException
   {
      Scanner scanner = new Scanner(new File(configFile));
      for (int numInput = 0; numInput < numInputVectors; numInput++)
      {
         for (int k = 0; k < layerSizes[0]; k++) // parses user input, assigns vals to units[0][k]
         {
            inputs[numInput][k] = scanner.nextDouble();
         }
         expectedValues[numInput] = scanner.nextDouble();
      }
   } // public void setInputs(String configFile) throws IOException

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
    * Calculates and returns the value of the final unit for the current training set. This method updates
    * the value of each
    * unit using the forwardUpdateUnit starting from the first hidden layer and ending with the
    * output layer. It then returns the calculated value of the final unit for the current training set.
    *
    * @param inputSet the input/output set that is currently being run
    *
    * @return the value of the final unit after calculation in the current training set.
    */
   public double calculateOutput(int inputSet)
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

      currentFinalValue = units[NUM_LAYERS - 1][0];
      return currentFinalValue; // the value of the only unit of the last layer (finalValue)
   } // public double calculateOutput(int inputSet)

   /**
    * Calculates the error of the perceptron based on the function E(x) = (x^2)/2 where x = the
    * difference between the value of the final input unit and the expected value. Uses the current
    * final value and current expected value that was calculated to calculate the error. Then, the method
    * returns the calculated error.
    *
    * A precondition for this method is that calculateOutput() has already been called and the
    * the final unit's value has already been calculated.
    *
    * @return the output of the error function of the perceptron.
    */
   public double calculateError()
   {
      double rawError = currentExpectedValue - currentFinalValue; // final value calculated by perceptron (output unit value)
      return (rawError * rawError) / 2.0;
   } // public double calculateError()

   /**
    * Prints all the values of the weights in the Perceptron at a given time.
    * The format in which the weights are printed is as follows:
    * w[0][0][0] = value0
    * w[0][0][1] = value1
    * w[0][1][0] = value2
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
    * @param outFile the file that this method outputs to.
    */
   public void updateWeights(String outFile) throws IOException
   {
      calculateDeltaWeights();

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
   } // public void updateWeights(String outFile) throws IOException

   /**
    * Calculates the adjustment values of each weight in the perceptron. This method does so
    * by calculating iterating through each weight connected to the final layer first, then
    * using a separate loop construct to iterate through remaining layers of the perceptron.
    * It then stores each value into an array, with one adjustment value for each weight.
    * Currently configured to run on an A-B-1 perceptron.
    */
   public void calculateDeltaWeights()
   {
      for (int k = 0; k < layerSizes[NUM_LAYERS - 2]; k++) // iterates through the weights in the final layer
      {
         deltaWeights[NUM_LAYERS - 2][k][0] = getFinalAdjustmentValue(k, 0); // sets adjustment values of final layer
         if (debug)
            System.out.println("DEBUG: Delta[" + (NUM_LAYERS - 2) + "][" + k + "][" + 0 + "]: " +
                  deltaWeights[NUM_LAYERS - 2][k][0]);
      }

      for (int n = NUM_LAYERS - 3; n >= 0; n--) // iterates through the hidden layers from right to left
      {
         for (int k = 0; k < layerSizes[n]; k++)
         {
            for (int j = 0; j < layerSizes[n + 1]; j++)
            {
               deltaWeights[n][k][j] = getHiddenAdjustmentValue(n, k, j); // sets the adjustment value of hidden layers
               if (debug)
               {
                  System.out.println("DEBUG: Delta[" + n + "][" + k + "][" + j + "]: " +
                        getHiddenAdjustmentValue(n, k, j));
               }

            } // for (int j = 0; j < layerSizes[n + 1]; j++)

         } // for (int k = 0; k < layerSizes[n]; k++)

      } // for (int n = NUM_LAYERS - 3; n >= 0; n--)

   } // public void calculateDeltaWeights()

   /**
    * The method that returns the adjustment value for the weights connected to the final layer of the perceptron.
    * Returns the value for a given weight that is identified by the previous and next units it is connected to.
    * This method is currently set to work in an A-B-1 network, so some code is specific to that configuration.
    * Calculates the value by multiplying the partial derivative of the error function with respect to the
    * current weight by the learning factor, as shown in the following expression:
    *
    * (−(T0 − F0)f'(netInputFinalLayer) * hj) * LEARNING_FACTOR
    *
    * where T0 is the expected value, F0 is the final value, and hj is the input unit of the weight.
    *
    * @param currentIndex the position of the unit that the weight is attached to in the input layer
    * @param nextIndex the position of the unit that the weight is attached to in the output layer
    * @return the adjustment value for the weight based on the current activation state
    */
   private double getFinalAdjustmentValue(int currentIndex, int nextIndex)
   {
      double partialDerivError = -1.0;

      double netInputPreviousLayer = 0.0;
      for (int k = 0; k < layerSizes[NUM_LAYERS - 2]; k++) // creates a net input based on previous layer
      {
         netInputPreviousLayer += units[NUM_LAYERS - 2][k] * weights[NUM_LAYERS - 2][k][nextIndex];
      }
      partialDerivError *= currentExpectedValue - currentFinalValue;
      partialDerivError *= activationFunctionDerivative(netInputPreviousLayer);
      partialDerivError *= units[NUM_LAYERS - 2][currentIndex];
      return -partialDerivError * learningFactor; // accumulator for the derivative eventually multiplied by learning factor
   }

   /**
    * The method that returns the adjustment value for the weight of any hidden layer of the perceptron.
    * Returns the value for a given weight that is identified by the previous and next units it is connected to
    * as well as the layer that the weight is in (layer number of the input unit)
    * Calculates the value by multiplying the partial derivative of the error function with respect to the
    * current weight by the learning factor, as shown in the following expression:
    *
    * (−ak * f'(netInputPreviousLayer) * (T0 − F0) * f'(netInputFinalLayer) * wj0) * LEARNING_FACTOR
    *
    * where T0 is the expected value, F0 is the final value, ak is the input unit of the weight, and
    * wj0 is the weight of the next layer that takes the current layer's output unit as its input unit.
    *
    * @param layerNum the layer that the weight is in
    * @param currentIndex the position of the unit that the weight is attached to in the input layer
    * @param nextIndex the position of the unit that the weight is attached to in the output layer
    * @return the adjustment value for the weight based on the current activation state
    */
   private double getHiddenAdjustmentValue(int layerNum, int currentIndex, int nextIndex)
   {
      double partialDerivError = -1.0;

      double netInputPreviousLayer = 0.0;
      for (int k = 0; k < layerSizes[layerNum]; k++) // creates a net input based on previous layer
      {
         netInputPreviousLayer += units[layerNum][k] * weights[layerNum][k][nextIndex];
      }

      double netInputNextLayer = 0.0;
      for (int k = 0; k < layerSizes[layerNum + 1]; k++) // creates a net input based on previous layer
      {
         netInputNextLayer += units[layerNum + 1][k] * weights[layerNum + 1][k][0]; // must update for multiple hidden layers
      }

      partialDerivError *= units[layerNum][currentIndex];
      partialDerivError *= activationFunctionDerivative(netInputPreviousLayer);
      partialDerivError *= currentExpectedValue - currentFinalValue;
      partialDerivError *= activationFunctionDerivative(netInputNextLayer);
      partialDerivError *= weights[NUM_LAYERS - 2][nextIndex][0];
      return -partialDerivError * learningFactor; // accumulator of the derivative eventually multiplied by learning factor
   }

   /**
    * Trains the perceptron by sequentially evaluating the perceptron for each input vector. After each
    * iteration, it updates the weights and outputs them to a given output file. One iteration consists of
    * calculating the output with one input/expected value set and, with the same set, updating the weights
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
         currentExpectedValue = expectedValues[inputNum];
         calculateOutput(inputNum);
         currentErrorVals[inputNum] = calculateError();
         updateWeights(outputFile);
      } // for (int inputNum = 0; inputNum < numInputVectors; inputNum++)
   } // public void run(String outputFile) throws IOException

   /**
    * Prints the outputs of the perceptron for each input vector on a new line by calculating it each time.
    */
   public void printOutputs()
   {
      for (int inputNum = 0; inputNum < numInputVectors; inputNum++)
      {
         System.out.println("Output " + inputNum + " is: " + calculateOutput(inputNum));
      }
   }

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
}
