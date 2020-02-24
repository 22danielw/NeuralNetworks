import java.util.Scanner;
import java.util.StringTokenizer;

/**
 * A class representing the A-B-1 perceptron. Contains functionality to
 * configure the number of units in the activation and hidden layer as
 * well as the number of hidden layers, setting the input values and initial weights
 * of the perceptron, and calculating the value of the final output.
 *
 * @author Daniel Wu
 * @version 02/13/2020
 * @version 2/24/2020
 */
public class Perceptron
{
   private final int NUM_LAYERS; // the number of layers in the perceptron, given by user
   private double[][] units; // the 2d array that represents the units of the perceptron
   private double[][][] weights; // the 3d array that represents the weights of the perceptron
   private double expectedValue; // the expected output of the perceptron
   private Scanner scanner = new Scanner(System.in); // scanner that takes in user input
   private boolean debug = true; // prints debug information if set to true

   /**
    * Constructs a A-B-1 perceptron, enabling the user to set the desired number of
    * hidden layers, the size of each layer except the output layer (which is set to 1).
    * The constructor also initializes the array of units and weights based on the user's
    * preferences (taken through the terminal). This constructor does not set the expected
    * value, values of input activations, or initial weights.
    */
   public Perceptron()
   {
      System.out.println("Please input number of desired number of HIDDEN layers");
      NUM_LAYERS = scanner.nextInt() + 2;
      units = new double[NUM_LAYERS][]; // creates the perceptron with desired number of layers

      System.out.println("Please input desired layer size for input layer");
      units[0] = new double[scanner.nextInt()]; // sets the input layer to size given by user

      for (int i = 1; i < NUM_LAYERS - 1; i++) // sets the each hidden layer to size given by user
      {
         System.out.println("Please input desired layer size for hidden layer " + i);
         units[i] = new double[scanner.nextInt()];
      }
      units[NUM_LAYERS - 1] = new double[1]; // sets the final output layer size to 1
      System.out.println("Output layer size has been set to 1 \n");

      weights = new double[NUM_LAYERS - 1][][];
      for (int i = 0; i < weights.length; i++) // creates weights array based on sizes of layers
      {
         int currentLayerSize = units[i].length;
         int nextLayerSize = units[i + 1].length;
         weights[i] = new double[currentLayerSize][nextLayerSize];
      }
   }

   /**
    * Sets the input activation values for the perceptron. The user should input
    * the values as prompted. The order in which these values should be inputted is provided.
    * These values are then set sequentially into the first layer of the perceptron, starting
    * from a[0][0] = first input, a[0][1] = second input, etc.. A precondition of this method
    * is that the user inputs the the values as prompted and inputs all the values necessary.
    */
   public void setValues()
   {
      String message = "Please input initial values in this order, separated by whitespaces:";
      System.out.println(message); // prompts reader for initial values

      for (int k = 0; k < units[0].length; k++) // prints the order that user should enter vals
      {
         System.out.println("a[0][" + k + "]");
      }

      scanner.nextLine(); // runs the cursor the scanner to the nextLine
      String input = scanner.nextLine();

      StringTokenizer inputTokenizer = new StringTokenizer(input);
      for (int k = 0; k < units[0].length; k++) // parses user input, assigns vals to units[][][]
      {
         double value = Double.parseDouble(inputTokenizer.nextToken(" "));
         units[0][k] = value;
      }
   }

   /**
    * Prompts the user for a double and sets it to the expected output of the perceptron. This
    * method stores the user input in an instance variable.
    */
   public void setExpectedValue()
   {
      System.out.println("Please input an expected value");
      expectedValue = scanner.nextDouble();
   }

   /**
    * Sets the weights for the perceptron. The user should input
    * the weights as prompted. The order in which these weights should be inputted is provided.
    * These weights are then set sequentially into the array of weights, where w[0][0][0] is the
    * first input, w[0][0][1] is the second input, w[0][1][0] is the third, etc..
    * A precondition of this method is that the user inputs the the weights as prompted and
    * inputs all the values necessary.
    */
   public void setWeights()
   {
      String m = "Please input initial weight values, separated by whitespaces in this order:";
      System.out.println(m); // message exceeds 100 characters

      for (int n = 0; n < weights.length; n++) // prints input order for weights
      {
         for (int j = 0; j < weights[n].length; j++)
         {
            for (int k = 0; k < weights[n][j].length; k++)
            {
               System.out.println("w[" + n + "][" + j + "][" + k + "]");
            }
         }
      }

      StringTokenizer weightTokenizer = new StringTokenizer(scanner.nextLine());

      for (int n = 0; n < weights.length; n++) // sets weights from user input into the array
      {
         for (int j = 0; j < weights[n].length; j++)
         {
            for (int k = 0; k < weights[n][j].length; k++)
            {
               weights[n][j][k] = Double.parseDouble(weightTokenizer.nextToken(" "));
            }
         }
      }
   }

   /**
    * Calculates and returns the value of the final unit. This method updates the value of each
    * unit using the forwardUpdateUnit starting from the first hidden layer and ending with the
    * output layer. It then returns the calculated value of the final unit.
    *
    * @return the value of the final unit after calculation.
    */
   public double calculateOutput()
   {
      for (int i = 1; i < NUM_LAYERS; i++) // updates units starting from the first hidden layer
      {
         for (int j = 0; j < units[i].length; j++)
         {
            forwardUpdateUnit(i, j);
         }
      }

      return units[units.length - 1][0]; // the value of the only unit of the last layer (final val)
   }

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
      double rawError = expectedValue -
            units[units.length - 1][0]; // final value calculated by perceptron (output unit value)
      return (rawError * rawError) / 2.0;
   }

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
         for (int j = 0; j < weights[n].length; j++)
         {
            for (int k = 0; k < weights[n][j].length; k++)
            {
               System.out.println("The value of w[" + n + "][" + j + "][" +
                     k + "] = " + weights[n][j][k]);
            }
         }
      }
   }

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
         for (int k = 0; k < units[n].length; k++)
         {
            System.out.println("The value of a[" + n + "][" + k + "] = " + units[n][k]);
         }
      }
   }

   /**
    * Updates a unit at a given location in a given layer. It adds takes the weighted
    * sum of all units in the previous layer, then takes an activation function on
    * the result and maps it to the unit at the given location. Also creates a debug
    * message and prints it if the program is set to debugging mode.
    *
    * A precondition for this method is that it is not called on layer 0 (input layer).
    *
    * @param layer the layer that the modified unit is in
    * @param location the location of the modified unit within its layer
    */
   private void forwardUpdateUnit(int layer, int location)
   {
      double netInput = 0.0;
      String debugMessage = "";
      for (int k = 0; k < units[layer - 1].length; k++) // creates a net input
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
      return 1 / (1 + Math.exp(-1 * input)); // sigmoid function
   }
}
