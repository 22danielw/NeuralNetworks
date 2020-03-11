import java.io.*;
import java.util.Scanner;
import java.util.StringTokenizer;

/**
 * A class representing the A-B-1 perceptron. Contains functionality to
 * configure the number of units in the activation and hidden layer as
 * well as the number of hidden layers, setting the input values and initial weights
 * of the perceptron, and calculating the value of the final output.
 *
 * @author Daniel Wu
 * @version 03/05/2020
 */
public class Perceptron
{
   private final int NUM_LAYERS; // the number of layers in the perceptron, given by user
   private final double LEARNING_FACTOR;
   private double[][] units; // the 2d array that represents the units of the perceptron
   private int[] layerSizes;
   private double[][][] weights; // the 3d array that represents the weights of the perceptron
   private double[][][] deltaWeights; //
   private boolean debug = false; // prints debug information if set to true
   private double[][] inputs;
   private double[] expectedValues;
   private double currentExpectedValue;
   private double currentFinalValue;
   private int numInputVectors;
   private double[] currentErrorVals;


   /**
    * The format of the structureConfig array is as follows:
    * [layerSize1, layerSize2, ... , layersizeN]
    * @param numLayers
    * @param structureConfig
    */
   public Perceptron(int numLayers, int[] structureConfig, int numInputVectors, double learningFactor)
   {
      NUM_LAYERS = numLayers;
      units = new double[NUM_LAYERS][]; // creates the perceptron with desired number of layers
      layerSizes = new int[NUM_LAYERS]; // creates the array storing layer sizes

      for (int n = 0; n < NUM_LAYERS; n++)
      {
         units[n] = new double[structureConfig[n]];
         layerSizes[n] = structureConfig[n];
      }

      weights = new double[NUM_LAYERS - 1][][];
      for (int i = 0; i < weights.length; i++) // creates weights array based on sizes of layers
      {
         int currentLayerSize = layerSizes[i];
         int nextLayerSize = layerSizes[i + 1];
         weights[i] = new double[currentLayerSize][nextLayerSize];
      }

      deltaWeights = new double[NUM_LAYERS - 1][][];
      for (int i = 0; i < deltaWeights.length; i++) // creates weights array based on sizes of layers
      {
         int currentLayerSize = layerSizes[i];
         int nextLayerSize = layerSizes[i + 1];
         deltaWeights[i] = new double[currentLayerSize][nextLayerSize];
      }

      inputs = new double[numInputVectors][];
      for (int n = 0; n < numInputVectors; n++)
      {
         inputs[n] = new double[layerSizes[0] + 1]; // the size of the initial activation layer + 1 for the expected value
      }
      expectedValues = new double[numInputVectors];
      this.numInputVectors = numInputVectors;

      LEARNING_FACTOR = learningFactor;
      currentErrorVals = new double[numInputVectors];
   } // public Perceptron()

   /**
    * The format of the config file is as follows where n is the number of input vectors,
    * a is the initial inputs, and e is the expected values.
    *
    * a11 a12
    * e1
    * a21 a22
    * e2
    * ..etc
    * This
    */
   public void setInputs(String configFile) throws IOException
   {
      Scanner scanner = new Scanner(new File(configFile));
      for (int numInput = 0; numInput < numInputVectors; numInput++)
      {
         for (int k = 0; k < layerSizes[0]; k++) // parses user input, assigns vals to units[][][]
         {
            inputs[numInput][k] = scanner.nextDouble();
         }
         expectedValues[numInput] = scanner.nextDouble();
      }
   } // public void setInputs(String configFile) throws IOException

   /**
    * Sets the weights for the perceptron. The user should input
    * the weights as prompted. The order in which these weights should be inputted is provided.
    * These weights are then set sequentially into the array of weights, where w[0][0][0] is the
    * first input, w[0][0][1] is the second input, w[0][1][0] is the third, etc..
    * A precondition of this method is that the user inputs the the weights as prompted and
    * inputs all the values necessary.
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
   }

   public void setWeights() throws IOException // default weight initialization
   {
      Scanner scanner = new Scanner(new File("weights.out"));
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
    * Calculates and returns the value of the final unit. This method updates the value of each
    * unit using the forwardUpdateUnit starting from the first hidden layer and ending with the
    * output layer. It then returns the calculated value of the final unit.
    *
    * @return the value of the final unit after calculation.
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
   } // public double calculateOutput()

   /**
    * Calculates the error of the perceptron based on the function E(x) = (x^2)/2 where x = the
    * difference between the value of the final input unit and the expected value.
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
    * <p>
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
    * adjustment value stored in the deltaWeights[] array.
    */
   public void updateWeights(String outFile) throws IOException
   {
      calculateDeltaWeights();
      PrintWriter writer = new PrintWriter(new File(outFile));
      for (int n = 0; n < NUM_LAYERS - 1; n++)
      {
         for (int k = 0; k < weights[n].length; k++)
         {
            for (int j = 0; j < weights[n][k].length; j++)
            {
               weights[n][k][j] += deltaWeights[n][k][j];
               writer.write(weights[n][k][j] + "\n");
            }
         }
      }
      writer.flush();
      writer.close();
   } // public void optimize()

   /**
    * Calculates the adjustment values of each weight in the perceptron. This method does so
    * by calculating iterating through each weight connected to the final layer first, then
    * using a separate loop construct to iterate through remaining layers of the perceptron.
    * It then stores each value into an array, with one adjustment value for each weight.
    * Currently configured to run on an A-B-1 perceptron.
    */
   public void calculateDeltaWeights()
   {
      for (int k = 0; k < weights[NUM_LAYERS - 2].length; k++)
      {
         deltaWeights[NUM_LAYERS - 2][k][0] = getFinalAdjustmentValue(k, 0);
         if (debug)
            System.out.println("DEBUG: Delta[" + (NUM_LAYERS - 2) + "][" + k + "][" + 0 + "]: " +
                  deltaWeights[NUM_LAYERS - 2][k][0]);
      }

      for (int layerNum = NUM_LAYERS - 3; layerNum >= 0; layerNum--)
      {
         for (int k = 0; k < weights[layerNum].length; k++)
         {
            for (int j = 0; j < weights[layerNum][k].length; j++)
            {
               deltaWeights[layerNum][k][j] = getHiddenAdjustmentValue(layerNum, k, j);
               if (debug)
               {
                  System.out.println("DEBUG: Delta[" + layerNum + "][" + k + "][" + j + "]: " +
                        getHiddenAdjustmentValue(layerNum, k, j));
               }

            } // for (int j = 0; j < weights[layerNum][k].length; j++)

         } // for (int k = 0; k < weights[layerNum].length; k++)

      } // for (int layerNum = NUM_LAYERS - 3; layerNum >= 0; layerNum--)

   } // calculateDeltaWeights()

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
      return partialDerivError * -1 * LEARNING_FACTOR;
   }

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
         netInputNextLayer += units[layerNum + 1][k] * weights[layerNum + 1][k][0]; // must update to accommodate multiple hidden layers
      }

      partialDerivError *= units[layerNum][currentIndex];
      partialDerivError *= activationFunctionDerivative(netInputPreviousLayer);
      partialDerivError *= currentExpectedValue - currentFinalValue;
      partialDerivError *= activationFunctionDerivative(netInputNextLayer);
      partialDerivError *= weights[NUM_LAYERS - 2][nextIndex][0];
      return partialDerivError * -1 * LEARNING_FACTOR;
   }

   public void run(String outputFile) throws IOException
   {
      for (int inputNum = 0; inputNum < numInputVectors; inputNum++)
      {
         currentExpectedValue = expectedValues[inputNum];
         calculateOutput(inputNum);
         currentErrorVals[inputNum] = calculateError();
         updateWeights(outputFile);
      }
   }

   public void printOutputs()
   {
      for (int i = 0; i < numInputVectors; i++)
      {
         System.out.println("Ouput " + i + " is: " + calculateOutput(i));
      }
   }

   public double getTotalError()
   {
      double totalError = 0.0;
      for (int inputNum = 0; inputNum < numInputVectors; inputNum++)
      {
         totalError += currentErrorVals[inputNum];
      }
      return totalError;
   }

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
      return 1.0 / (1.0 + Math.exp(-1.0 * input)); // sigmoid function
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
