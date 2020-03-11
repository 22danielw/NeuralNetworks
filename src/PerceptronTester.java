import java.io.IOException;

/**
 * A tester class that tests the basic functionality of the perceptron.
 * Prints information relevant to debugging and testing such as the output and
 * error of the calculation.
 *
 * @author Daniel Wu
 * @version 2/24/2020
 */
public class PerceptronTester
{
   /**
    * Testing main method that provides an interface to test a Perceptron.
    * The perceptron can be tested as many times as the user wants. Prints
    * the output and error at the end of execution, and uses the same perceptron
    * object to run all the tests.
    *
    * The runtime arguments are formatted as follows:
    *
    * numLayers L1 L2 L3 ... LN numInputs learningFactor
    *
    * @param args Runtime arguments
    */
   public static void main(String args[]) throws IOException
   {
      int numLayers = Integer.parseInt(args[0]);
      int[] structureConfig = new int[numLayers];
      for (int n = 0; n < numLayers; n++)
      {
         structureConfig[n] = Integer.parseInt(args[n + 1]);
      }
      int numInputs = Integer.parseInt(args[numLayers + 1]);
      double learningFactor = Double.parseDouble(args[numLayers + 2]);
      Perceptron perceptron = new Perceptron(numLayers, structureConfig, numInputs, learningFactor);

      perceptron.setInputs("inputFile.in");
      perceptron.setWeights();
      perceptron.run("weights.out");
      int iterationCount = 1;

      while (iterationCount < 1000 && perceptron.getTotalError() > .001)
      {
         perceptron.run("weights.out");
         iterationCount++;
         System.out.println("Error is: " + perceptron.calculateError());
      }
      System.out.println("Error is: " + perceptron.calculateError() + "\n");
      perceptron.printOutputs();
      System.out.println("Perceptron converged in " + iterationCount + " iterations.");

   } // public static void main(String args[])
}
