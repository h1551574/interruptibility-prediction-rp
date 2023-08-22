package org.interruption_experiment;// Importing required classes


import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.ZeroR;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

public class PerInstanceValidation5StateTWs {

    public static void main(String[] args) {
        String[] timeWindows = {
                "10",
                "20",
                "30",
                "45",
                "60",
                "120",
                "180",
        };
        for (String tw : timeWindows){
            RepeatedCrossValidation(tw);
        }
    }
    public static void RepeatedCrossValidation(String timeWindow) {
        System.out.println("TW: "+timeWindow);
        // Try block to check for exceptions
        try {


            // Dataset path
            String InterruptionDataset
                    = "C:/Users/alerr/Documents/Masterarbeit/Normalized Data/"+timeWindow+"/with_pID/" +
                    "normalized_interruption_data_ALL_tw_"+timeWindow+"s_pID_3hz_5state_filtered_attributes.arff";


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





                    average_kappa += eval.kappa();
                    Evaluation eval_zr = new Evaluation(train);
                    ZeroR zr = new ZeroR();
                    zr.buildClassifier(train);
                    eval_zr.evaluateModel(zr,test);

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
                    }
                }
            }


 //           System.out.println("----------------------------END RESULT----------------------------");



            StandardDeviation sd = new StandardDeviation();
            System.out.println("[NB] Avg. Accuracy: " + Utilites.calculateMean(accuraciesNB));
            System.out.println("[NB] Std. Accuracy: "+ sd.evaluate(accuraciesNB));

        }

        // Catch block to handle the exceptions
        catch (Exception e) {

            // Print message on the console
            System.out.println("Error Occurred!!!! \n"
                    + e.getMessage());
        }
    }



}