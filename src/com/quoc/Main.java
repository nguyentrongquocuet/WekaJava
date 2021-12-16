package com.quoc;

import autoweka.Configuration;
import autoweka.ConfigurationCollection;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AutoWEKAClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

class AWK extends  AutoWEKAClassifier {
    public Classifier getChosenClassifier() {
        return classifier;
    }

    public String getClassifierClass() {
        return classifierClass;
    }

    public ConfigurationCollection getBestConfigurationCollection() {
        return bestConfigsCollection;
    }
}

public class Main {
    static int NTIME_LIMIT = 3;
    static String TIME_LIMIT = Integer.toString(NTIME_LIMIT);
    static int NSEED = 42;
    static String SEED = Integer.toString(NSEED);
    static int NNBEST_CONFS = 3;
    static String N_BEST_CONFS = Integer.toString(NNBEST_CONFS);
    static int NMEM_LIMIT = 3000;
    static String MEM_LIMIT = Integer.toString(NMEM_LIMIT);
    static int NPARALLEL_RUN = 2;
    static String PARALLEL_RUN = Integer.toString(NPARALLEL_RUN);

    static void breathe(int index, Writer resultWriter) throws Exception {
        breathe(index, NSEED, resultWriter);
    }

    static void breathe(int index) throws Exception {
        breathe(index, NSEED, null);
    }

    /**
     *
     * @param index number respresent train-test pair
     * @param seed seed (random_state in other word)
     * @param resultWriter a writer will be used for saving result
     * @throws Exception
     */
    static void breathe(int index, int seed, Writer resultWriter) throws Exception {
        String trainPath = "datasets/splitted/train-" + (index + 1) + ".arff";
        DataSource trainSource = new DataSource(trainPath);
        Instances  trainData = trainSource.getDataSet();
        trainData.setClassIndex(trainData.numAttributes() - 1);

        String testPath = "datasets/splitted/test-" + (index + 1) + ".arff";
        DataSource testSource = new DataSource(testPath);
        Instances  testData = testSource.getDataSet();
        testData.setClassIndex(testData.numAttributes() - 1);

        AWK classifier = new AWK();

        String[] autoWekaOptions = { "-seed", SEED, "-timeLimit", TIME_LIMIT, "-memLimit", MEM_LIMIT, "-nBestConfigs", N_BEST_CONFS, "-metric", "errorRate", "-parallelRuns", PARALLEL_RUN };
        classifier.setOptions(autoWekaOptions);

        classifier.buildClassifier(trainData);
        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(classifier, testData);
        String info = eval.toClassDetailsString("\nResults\n======\n");
        String no1Classifier = classifier.getClassifierClass();
        Double f1 = eval.weightedFMeasure();
        Double recall = eval.weightedRecall();
        Double precicion = eval.weightedPrecision();
        Double accuracy = eval.pctCorrect() ;
        System.out.println(info);
        System.out.println(classifier);
        /**
         * Get all configurations
         */
        ConfigurationCollection c = classifier.getBestConfigurationCollection();
        List<String> parsedConfigurations = new ArrayList<>();
        if (c != null) {
            ArrayList<Configuration> configList = c.asArrayList();
            for (Configuration configuration : configList) {
                String classifierName = configuration.getArgStrings().split("-targetclass ")[1].trim();
                parsedConfigurations.add(classifierName);
            }
        }

        String no2Classifier = parsedConfigurations.get(1).toString();
        String no3Classifier = parsedConfigurations.get(2).toString();
        if (resultWriter == null) return;
        /**
         * numOfTry    no_1   	f1  	precision	recall	accuracy	no_2	no_3	seed
         */
        String lineToWrite = String.format("%s, %s, %s, %s, %s, %s, %s, %s, %s\n", index+1, no1Classifier, f1, precicion, recall, accuracy, no2Classifier, no3Classifier, seed);
        resultWriter.write(lineToWrite);
        resultWriter.flush();
    }

    static void prodRun() {
        prodRun(30);
    }

    static void prodRun(int numOfTry) {
        prodRun(numOfTry, new int[0]);
    }

    static void prodRun(int numOfTry, int[] seedList) {
        try {
            String date = new Date().toString().replaceAll(" ", "");
            FileWriter resultFile = new FileWriter("results/" + date + ".csv");
            BufferedWriter resultWriter = new BufferedWriter(resultFile);
            resultWriter.write("numOfTry, no_1, f1, precision, recall, accuracy, no_2, no_3, seed\n");
            resultWriter.flush();
            int l = seedList.length;
            for (int i=0; i< numOfTry; i++) {
                int seed = i < l ? seedList[i] : NSEED;
                breathe(i, seed, resultWriter);
            }

            resultWriter.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    static void testRun() {
        try {
            int[] arr = { 7, 17, 27 };
            prodRun(3, arr);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
//        testRun();
        prodRun(30);
    }
}
