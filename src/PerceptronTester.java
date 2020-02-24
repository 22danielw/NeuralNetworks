import java.util.Scanner;

/**
 * A tester class that tests the basic functionality of the perceptron.
 * Prints information relevant to debugging and testing such as the output and
 * error of the calculation.
 *
 * @author Daniel Wu
 * @version 2/13/2020
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
    * @param args Runtime arguments
    */
   public static void main(String args[])
   {
      Scanner scanner = new Scanner(System.in);
      Perceptron perceptron = new Perceptron();
      String prompt = "p"; // sets prompt to "p" for initial testing
      while (prompt.equals("p")) // while the user enters "p", the perceptron keeps running
      {
         perceptron.setValues();
         perceptron.setWeights();
         perceptron.setExpectedValue();
         System.out.println("\nOUTPUT = " + perceptron.calculateOutput());
         System.out.println("\nERROR = " + perceptron.calculateError());
         System.out.println("\nPlease type p to proceed with Perceptron testing");
         prompt = scanner.nextLine();
      } // while (prompt.equals("p"))
   }
}
