import java.io.File;
import java.io.IOException;
import java.util.Scanner;

/**
 * A tester class that tests the basic functionality of the perceptron.
 * Prints information relevant to debugging and testing such as the output,
 * error of the calculation, and various hyper-parameters. The tester also tests
 * the training ability of the perceptron.
 *
 * @author Daniel Wu
 * @version 2/24/2020
 * @version 03/23/2020
 */
public class PerceptronTester
{
   private static int MAX_ITERATIONS;
   private static double ERROR_THRESHOLD;

   /**
    * Testing main method that provides an interface to test a Perceptron.
    * The perceptron can be tested as many times as the user wants. Prints
    * the output and error at the end of execution, and uses the same perceptron
    * object to run all the tests. Saves the weights after each run and at the end.
    *
    * IMPORTANT FOR CONFIGURATION: If the weightConfigFileName given by the user is "RNGWeights", random
    * weights generated from the upper and lower bounds provided in the config file will be used. If the user
    * wants to use their own initial weights, please do not name it "RNGWeights"
    *
    * The runtime arguments are formatted as follows:
    *
    * configFileName numInputs inputFileName outputFileName weightConfigFileName
    *
    * The configuration file format is as follows:
    *
    * numLayers
    * L1 L2 L3 ... LN (size of each layer)
    * lambdaLearningFactor
    * maxIterations
    * errorThreshold
    * lowerWeightLimit upperWeightLimit
    *
    * The tester then trains the perceptron until the max iteration count allowed is reached or the perceptron's
    * total error goes below the threshold. Then, the tester prints the number of iterations it took
    * to reach the result, the output and the error.
    *
    *
    * @param args Runtime arguments containing information to initialize and configure the perceptron.
    */
   public static void main(String args[]) throws IOException
   {
      Scanner scanner = new Scanner(new File(args[0]));
      int numLayers = scanner.nextInt();              // sets number of layers
      int[] layerSizes = new int[numLayers];
      for (int i = 0; i < numLayers; i++)             // sets perceptron layer sizes
      {
         layerSizes[i] = scanner.nextInt();
      }
      double learningFactor = scanner.nextDouble();   // sets learning factor
      MAX_ITERATIONS = scanner.nextInt();             // sets the max iterations
      ERROR_THRESHOLD = scanner.nextDouble();         // sets error threshold
      double lowerWeightLimit = scanner.nextDouble(); // sets the lower limit of the random weight range
      double upperWeightLimit = scanner.nextDouble(); // sets the upper limit of the random weight range

      int numInputs = Integer.parseInt(args[1]);
      String inputFileName = args[2];
      String outputFileName = args[3];
      String weightInputFileName = args[4];

      Perceptron perceptron = new Perceptron(numLayers, layerSizes, numInputs, learningFactor);

      perceptron.setInputs(inputFileName);     // initializes inputs for the weight
      if (weightInputFileName.equals("RNGWeights")) // sets the weights either randomly or from a file based on user preference
      {
         perceptron.setWeights(lowerWeightLimit, upperWeightLimit);
      }
      else
      {
         perceptron.setWeights(weightInputFileName);
      }
      perceptron.train(outputFileName);
      int iterationCount = 1;

      while (iterationCount < MAX_ITERATIONS && perceptron.getTotalError() > ERROR_THRESHOLD)
      {
         perceptron.train(outputFileName);
         iterationCount++;
      }

      if (perceptron.getTotalError() > ERROR_THRESHOLD)
         System.out.println("Training ended by reaching max iterations allowed.");
      else
         System.out.println("Training ended by error going below error threshold.");

      System.out.println("Error threshold was: " + ERROR_THRESHOLD); // printing hyper-parameters and debug info
      System.out.println("Random weight range was from " + lowerWeightLimit + " to " + upperWeightLimit);

      for (int n = 0; n < perceptron.getLayerSizes().length; n++)
      {
         System.out.println("Layer " + n + " has " + layerSizes[n] + " activations");
      }

      System.out.println("Starting learning factor was " + learningFactor);
      System.out.println("Final learning factor was " + perceptron.getLearningFactor() + "\n");
      System.out.println("Max iterations allowed was " + MAX_ITERATIONS + "\n");


      System.out.println("Total error reached is: " + perceptron.getTotalError());
      perceptron.printOutputs();
      System.out.println("Perceptron converged in " + iterationCount + " iterations.");

   } // public static void main(String args[]) throws IOException
}
