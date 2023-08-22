package org.interruption_experiment;

import static java.lang.Math.sqrt;

public class Utilites {
    public static double calculateStandardDeviation(double[] array) {

        // get the sum of array
        double sum = 0.0;
        for (double i : array) {
            sum += i;
        }

        // get the mean of array
        int length = array.length;
        double mean = sum / length;

        // calculate the standard deviation
        double standardDeviation = 0.0;
        for (double num : array) {
            standardDeviation += Math.pow(num - mean, 2);
        }

        return sqrt(standardDeviation / length);
    }

    public static double calculateMean(double[] array) {
        // getting array length
        int length = array.length;

        // default sium value.
        int sum = 0;

        // sum of all values in array using for loop
        for (int i = 0; i < array.length; i++) {
            sum += array[i];
        }

        return sum / length;
    }
}
