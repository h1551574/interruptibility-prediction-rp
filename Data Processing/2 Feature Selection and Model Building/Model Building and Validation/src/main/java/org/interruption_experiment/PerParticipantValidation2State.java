package org.interruption_experiment;


import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.stat.inference.TTest;
import org.apache.commons.statistics.distribution.TDistribution;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.ZeroR;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;
import java.util.Random;

import static java.lang.Math.sqrt;

public class PerParticipantValidation2State {


    public static void main(String[] args) {
        System.out.println("Hello world!");
        // Try block to check for exceptions
        try {
            // Dataset path
            //String InterruptionDataset
            //        = "C:/Users/alerr/Documents/Masterarbeit/Normalized Data/10/with_pID/normalized_interruption_data_ALL_tw_10s_pID_3hz_2state_filtered_attributes.arff";


            int NUM_PARTICIPANTS = 10;
            Instances[] participants_train = new Instances[NUM_PARTICIPANTS];
            Instances[] participants_test = new Instances[NUM_PARTICIPANTS];
            for (int pi = 1; pi <= NUM_PARTICIPANTS; pi++) {
                String TrainDataset
                        = "C:/Users/alerr/Documents/Masterarbeit/Normalized Data/10/with_pID/CV LOPO/" +
                        "norm_inter_data_ALL_10s_pID_3hz_2state_filtered_attributes_train_P"+ pi +".arff";

                // Creating bufferedreader to read the dataset
                BufferedReader trainBufferedReader
                        = new BufferedReader(
                        new FileReader(TrainDataset));

                // Create dataset instances
                participants_train[pi-1]
                        = new Instances(trainBufferedReader);

                String TestDataset
                        = "C:/Users/alerr/Documents/Masterarbeit/Normalized Data/10/with_pID/CV LOPO/" +
                        "norm_inter_data_ALL_10s_pID_3hz_2state_filtered_attributes_test_P"+ pi +".arff";

                // Creating bufferedreader to read the dataset
                BufferedReader testBufferedReader
                        = new BufferedReader(
                        new FileReader(TestDataset));

                // Create dataset instances
                participants_test[pi-1]
                        = new Instances(testBufferedReader);

            }

            // Set Target Class
            for (Instances trainInstances : participants_train) {
                trainInstances.setClassIndex(
                        trainInstances.numAttributes() - 1);
            }
            for (Instances testInstances : participants_test){
                testInstances.setClassIndex(
                        testInstances.numAttributes() - 1);
            }
            int NUM_FOLDS = 10;
            int NUM_RUNS = 10;
            double[][] summedConfusionMatrix = {{0, 0}, {0, 0}};
            Random rnd = new Random(1551574);

            double average_kappa = 0;
            double[] accuraciesNB = new double[NUM_PARTICIPANTS];
            double[] accuraciesZR = new double[NUM_PARTICIPANTS];
            double totalTrainSize = 0.0;
            double totalTestSize = 0.0;
            double totalPerformanceDifference = 0.0;
            double[] performanceDifferences = new double[NUM_PARTICIPANTS];
            double totalAccuracyNB = 0.0;
            double totalAccuracyZR = 0.0;


            for (int pi = 0; pi < NUM_RUNS; pi++) {
                Instances train = participants_train[pi];
                Instances test = participants_test[pi];

                //Naive Bayes
                NaiveBayes nb = new NaiveBayes();
                nb.buildClassifier(train);
                Evaluation evalNB = new Evaluation(train);
                evalNB.evaluateModel(nb, test);


                // then I compute each folds' results using eval.XXX()
                System.out.println("Left Out Participant: " + Integer.toString(pi+1));
                System.out.println(evalNB.toSummaryString());

                System.out.println(Arrays.toString(evalNB.confusionMatrix()[0]));
                System.out.println(Arrays.toString(evalNB.confusionMatrix()[1]));
                average_kappa += evalNB.kappa();
                accuraciesNB[pi] = evalNB.pctCorrect();
                Evaluation evalZR = new Evaluation(train);
                ZeroR zr = new ZeroR();
                zr.buildClassifier(train);
                evalZR.evaluateModel(zr, test);
                System.out.println("ZeroR Accuracy: " + evalZR.pctCorrect());

                accuraciesZR[pi] = evalZR.pctCorrect();

                totalTrainSize += train.size();
                totalTestSize += test.size();
                totalPerformanceDifference += evalNB.pctCorrect() - evalZR.pctCorrect();
                performanceDifferences[pi] = evalNB.pctCorrect() - evalZR.pctCorrect();

                for (int ci = 0; ci < 2; ci++) {
                    for (int cj = 0; cj < 2; cj++) {
                        summedConfusionMatrix[ci][cj] = summedConfusionMatrix[ci][cj] + evalNB.confusionMatrix()[ci][cj];    //use - for subtraction
                    }
                }
            }

            System.out.println("----------------------------END RESULT----------------------------");

            System.out.println("Summed Confusion Matrix:");
            System.out.println(Arrays.toString(summedConfusionMatrix[0]));
            System.out.println(Arrays.toString(summedConfusionMatrix[1]));
            double TP = summedConfusionMatrix[0][0];
            double TN = summedConfusionMatrix[1][1];
            double FP = summedConfusionMatrix[0][1];
            double FN = summedConfusionMatrix[1][0];
            double n = TP + TN + FP + FN;
            double accuracy = (TP + TN) / n;
         //   System.out.println("[NB] Avg. Accuracy: " + accuracy * 100);
         //   System.out.println(Arrays.toString(accuraciesNB));
            StandardDeviation sd = new StandardDeviation();
            System.out.println("[NB] Avg. Accuracy: " + Utilites.calculateMean(accuraciesNB));
            System.out.println("[NB] Std. Accuracy: "+ sd.evaluate(accuraciesNB));
            System.out.println("Avg. Kappa: " + (average_kappa / (NUM_PARTICIPANTS)));

            System.out.println("[ZR] Avg. Accuracy: " + Utilites.calculateMean(accuraciesZR));
            System.out.println("[ZR] Std. Accuracy: " + Utilites.calculateStandardDeviation(accuraciesZR));


            // Corrected T-Test
            // Source: Bouckaert, Remco R., and Eibe Frank. "Evaluating the replicability of significance tests for comparing learning algorithms." PAKDD. Vol. 3056. 2004.
            double testTrainRatio = 0.0;
            testTrainRatio = totalTestSize / totalTrainSize;
            System.out.println("Test Train Ratio:" + testTrainRatio);
            System.out.println("performance diff:" + totalPerformanceDifference);

            double kr = ((double) NUM_PARTICIPANTS);
            double m = (1 / kr) * totalPerformanceDifference;
            double estiamated_variance = 0.0;
            for (int pi = 0; pi < NUM_RUNS; pi++) {
                    estiamated_variance += (1 / (kr - 1)) * Math.pow((performanceDifferences[pi] - m), 2);
            }
            double numerator = m;
            double denominator = sqrt(((1 / kr) + (testTrainRatio)) * estiamated_variance);
            double t = numerator / denominator;
            double df = kr - 2;

            TDistribution tDistribution = TDistribution.of(df);
            double p = (1 - tDistribution.cumulativeProbability(t)) * 2;

            TTest tTest = new TTest();
            System.out.println("P (Simple-T-Test): "+tTest.pairedTTest(accuraciesNB,accuraciesZR));

            System.out.println("T:" + t);
            System.out.println("DF: " + df);
            System.out.println("P (T-Test Corrected):" + p);
            double majorityClassifierAccuracy = (50.0/82.0)*100.0;
            System.out.println("Majority Classifier Accuracy: "+majorityClassifierAccuracy);
            System.out.println("P (Simple-One-Sample-T-Test (vs. Majority): "+tTest.tTest(majorityClassifierAccuracy,accuraciesNB));



        }

        // Catch block to handle the exceptions
        catch (Exception e) {

            // Print message on the console
            System.out.println("Error Occurred!!!! \n"
                    + e.getMessage());
            e.printStackTrace();
        }


    }
}


