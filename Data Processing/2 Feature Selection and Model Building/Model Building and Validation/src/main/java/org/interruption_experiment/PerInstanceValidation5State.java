package org.interruption_experiment;// Importing required classes

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

public class PerInstanceValidation5State {

    public static void main(String[] args) {
        System.out.println("Hello world!");
        // Try block to check for exceptions
        try {


            // Dataset path
            String InterruptionDataset
                    = "C:/Users/alerr/Documents/Masterarbeit/Normalized Data/10/with_pID/" +
                    "normalized_interruption_data_ALL_tw_10s_pID_3hz_5state_filtered_attributes.arff";


            // Creating bufferedreader to read the dataset
            BufferedReader bufferedReader
                    = new BufferedReader(
                    new FileReader(InterruptionDataset));

            // Create dataset instances
            Instances datasetInstances
                    = new Instances(bufferedReader);

            // Set Target Class
            datasetInstances.setClassIndex(
                    datasetInstances.numAttributes() - 1);
            int NUM_FOLDS = 10;
            int NUM_RUNS = 10;
            int NUM_CATEGORIES = 5;

            double[][] summedConfusionMatrix ={
                    {0,0,0,0,0},
                    {0,0,0,0,0},
                    {0,0,0,0,0},
                    {0,0,0,0,0},
                    {0,0,0,0,0}
            };
            Random rnd = new Random(1551574);

            double average_kappa = 0;
            double[] accuraciesNB = new double[NUM_FOLDS*NUM_RUNS];
            double[] accuraciesZR = new double[NUM_FOLDS*NUM_RUNS];
            int counter = 0;
            double totalTrainSize = 0.0;
            double totalTestSize = 0.0;
            double totalPerformanceDifference = 0.0;
            double[][] performanceDifferences = new double[NUM_RUNS][NUM_FOLDS];

            for(int ri=0; ri<NUM_RUNS; ri++){
                datasetInstances.randomize(rnd);
                datasetInstances.stratify(NUM_FOLDS); // stratify the dataset into 10 folds

                for(int fi=0; fi<NUM_FOLDS; fi++){
                    Instances train = datasetInstances.trainCV(NUM_FOLDS, fi);
                    Instances test = datasetInstances.testCV(NUM_FOLDS, fi);
                    //Naive Bayes
                    NaiveBayes nb = new NaiveBayes();
                    nb.buildClassifier(train);
                    Evaluation eval = new Evaluation(train);
                    eval.evaluateModel(nb, test);



                    // then I compute each folds' results using eval.XXX()
                    System.out.println("Fold: "+ Integer.toString(fi));
                    System.out.println(eval.toSummaryString());

               //     System.out.println(Arrays.toString(eval.confusionMatrix()[0]));
               //     System.out.println(Arrays.toString(eval.confusionMatrix()[1]));
                    average_kappa += eval.kappa();
                    Evaluation eval_zr = new Evaluation(train);
                    ZeroR zr = new ZeroR();
                    zr.buildClassifier(train);
                    eval_zr.evaluateModel(zr,test);
                    System.out.println("ZeroR Accuracy: "+eval_zr.pctCorrect());
                    System.out.println("NB Accuracy: "+eval.pctCorrect());

                    totalTrainSize += train.size();
                    totalTestSize += test.size();
                    totalPerformanceDifference += eval.pctCorrect() - eval_zr.pctCorrect();
                    performanceDifferences[ri][fi] = eval.pctCorrect() - eval_zr.pctCorrect();


                    accuraciesNB[counter] =  eval.pctCorrect();
                    accuraciesZR[counter] =  eval_zr.pctCorrect();

                    counter += 1;
                    for(int ci=0;ci<NUM_CATEGORIES;ci++){
                        for(int cj=0;cj<NUM_CATEGORIES;cj++){
                            summedConfusionMatrix[ci][cj]=summedConfusionMatrix[ci][cj]+eval.confusionMatrix()[ci][cj];    //use - for subtraction
                        }
                        System.out.println(Arrays.toString(eval.confusionMatrix()[ci]));
                    }
                }
            }


            System.out.println("----------------------------END RESULT----------------------------");

            System.out.println("Summed Confusion Matrix:");
            for(int ci=0;ci<NUM_CATEGORIES;ci++){
                for(int cj=0;cj<NUM_CATEGORIES;cj++){
                    summedConfusionMatrix[ci][cj]=summedConfusionMatrix[ci][cj]/10;    //use - for subtraction
                }
                System.out.println(Arrays.toString(summedConfusionMatrix[ci]));
            }

            StandardDeviation sd = new StandardDeviation();
            System.out.println("[NB] Avg. Accuracy: " + Utilites.calculateMean(accuraciesNB));
            System.out.println("[NB] Std. Accuracy: "+ sd.evaluate(accuraciesNB));
            System.out.println("Avg. Kappa: "+ (average_kappa / (NUM_FOLDS * NUM_RUNS)));

            System.out.println("[ZR] Avg. Accuracy: " + Utilites.calculateMean(accuraciesZR));
            System.out.println("[ZR] Std. Accuracy: "+ Utilites.calculateStandardDeviation(accuraciesZR));
            // Corrected T-Test
            // Source: Bouckaert, Remco R., and Eibe Frank. "Evaluating the replicability of significance tests for comparing learning algorithms." PAKDD. Vol. 3056. 2004.
            double testTrainRatio = 0.0;
            testTrainRatio = totalTestSize / totalTrainSize;
            System.out.println("Test Train Ratio:"+ testTrainRatio);
            System.out.println("performance diff:"+ totalPerformanceDifference);

            double kr = ((double) NUM_FOLDS *(double) NUM_RUNS);
            double m = ( 1 / kr)*totalPerformanceDifference;
            double estiamated_variance = 0.0;
            for(int ri=0; ri<NUM_RUNS; ri++) {
                for (int fi = 0; fi < NUM_FOLDS; fi++) {
                    estiamated_variance += (1/(kr-1))*Math.pow((performanceDifferences[ri][fi]-m),2);
                }
            }
            double numerator = m;
            double denominator = sqrt(((1/kr)+(testTrainRatio))*estiamated_variance);
            double t = numerator / denominator;
            double df = kr-2;

            TDistribution tDistribution = TDistribution.of(df);
            double p = (1 - tDistribution.cumulativeProbability(t))*2;


            System.out.println("T:" + t);
            System.out.println("DF: " + df);
            System.out.println("P (T-Test Corrected):" + p);
            TTest tTest = new TTest();
            double majorityClassifierAccuracy = (29.0/82.0)*100.0;
            System.out.println("Majority Classifier Accuracy: "+majorityClassifierAccuracy);
            System.out.println("P (Simple-One-Sample-T-Test (vs. Majority): "+tTest.tTest(majorityClassifierAccuracy,accuraciesNB));




        }

        // Catch block to handle the exceptions
        catch (Exception e) {

            // Print message on the console
            System.out.println("Error Occurred!!!! \n"
                    + e.getMessage());
        }
    }



}