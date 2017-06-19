package com.spbsu.exp.greedyRegions;

import com.spbsu.commons.random.FastRandom;
import com.spbsu.crawl.sessions.WeightedRandomWalkGameSession;
import com.spbsu.exp.cart.CARTSteinEasy;
import com.spbsu.exp.cart.DataLoader;
import com.spbsu.exp.cart.Utils;
import com.spbsu.exp.cart.VotedInterval;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.LOOL2;
import com.sun.xml.internal.ws.util.StreamUtils;
import javafx.scene.shape.Path;

import javax.xml.crypto.Data;
import java.awt.*;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.stream.Stream;

import static com.spbsu.exp.cart.DataLoader.bootstrap;
import static com.spbsu.exp.cart.DataLoader.readData;

/**
 * Created by nadya-bu on 12/06/2017.
 */
public class TestGreedyTDSimpleRegion {
    private static final String TestBaseDataName = "featuresTest.txt.gz";
    private static final String LearnBaseDataName = "features.txt.gz";
    private static final String LearnCTSliceFileName = "slice_train.csv";
    private static final String TestCTSliceFileName = "slice_test.csv";
    private static final String LearnKSHouseFileName = "learn_ks_house.csv";
    private static final String TestKSHouseFileName = "test_ks_house.csv";
    private static final String LearnHIGGSFileName = "HIGGS_learn_1M.csv.gz";
    private static final String TestHIGGSFileName = "HIGGS_test.csv.gz";
    private static final String LearnCancerFileName = "Cancer_learn.csv";
    private static final String TestCancerFileName = "Cancer_test.csv";


    private static String PATH_TO_DATA;
    private static final Integer NUM_ITER = 100;

    @FunctionalInterface
    interface Optimization<Data, NumIter, Step, Loss, RegCoeff, Score> {
        Score apply(Data data, NumIter numIter, Step step, Loss loss, RegCoeff e, String logFile);
    }

    private static void runExperimentScore(DataLoader.DataFrame data,
                                      Optimization<DataLoader.DataFrame, Integer, Double, Class, Double, Double> evaluate,
                                      int iter, double step,
                                      Class<?> loss, double regCoeff, String fileName, String prefix) throws IOException {

        System.out.println(fileName);

        new Thread(() -> {

            File file = new File(fileName);
            PrintWriter writer;
            try {
                writer = new PrintWriter(new FileWriter(file));
            } catch (IOException e) {
                System.out.println("Could not create file to write log: " + file);
                return;
            }
            final int numIter = NUM_ITER;
            DataLoader.DataFrame cur_data;
            for (int i = 0; i < numIter; i++) {
                cur_data = bootstrap(data, i);
                String logFile = Paths.get(prefix + "seed" + Long.toString(i) + ".txt").toString();
                System.out.printf("Iteration No %d\n", i);
                double score = evaluate.apply(cur_data, iter, step, loss, regCoeff, logFile);
                writer.write(Double.toString(score) + "\n");
                writer.flush();
            }

            writer.close();

        }).start();
    }

    private static void runExperimentInterval(DataLoader.DataFrame data,
                                           Optimization<DataLoader.DataFrame, Integer, Double, Class, Double, Double> evaluate,
                                           int iter, double step,
                                           Class<?> loss, double regCoeff, String fileName, String prefix) throws IOException {

        System.out.println(fileName);

        new Thread(() -> {

            File file = new File(fileName);
            PrintWriter writer;
            try {
                writer = new PrintWriter(new FileWriter(file));
            } catch (IOException e) {
                System.out.println("Could not create file to write log: " + file);
                return;
            }
            final int numIter = NUM_ITER;
            double scores[] = new double[numIter];
            Stream.iterate(0, i -> i + 1).limit(numIter).parallel().forEach((i) -> {
                final long seed = i;//new FastRandom(i).nextLong();
                String logFile = prefix + "seed" + Long.toString(seed) + ".txt";
                DataLoader.DataFrame cur_data = bootstrap(data, seed);
                scores[i] = evaluate.apply(cur_data, iter, step, loss, regCoeff, logFile);
                synchronized (writer) {
                    writer.write(String.format("Iteration No %d, score = %.7f\n", i, scores[i]));
                    writer.flush();
                }
            });
            Arrays.sort(scores);
            writer.write("[" + Double.toString(scores[5]) + ", " + Double.toString(scores[numIter - 6])  + "]\n");
            writer.close();
        }).start();
    }

    private static void runTest(DataLoader.DataFrame data,
                                              Optimization<DataLoader.DataFrame, Integer, Double, Class, Double, Double> evaluate,
                                              int iter, double step,
                                              Class<?> loss, double regCoeff, String fileName, String prefix) throws IOException {

        System.out.println(fileName);

        new Thread(() -> {

            File file = new File(fileName);
            PrintWriter writer;
            try {
                writer = new PrintWriter(new FileWriter(file));
            } catch (IOException e) {
                System.out.println("Could not create file to write log: " + file);
                return;
            }
            final int numIter = NUM_ITER;
            DataLoader.DataFrame cur_data;
            cur_data = bootstrap(data, System.currentTimeMillis());
            String logFile = prefix +"testLog.txt";
            double score = evaluate.apply(cur_data, iter, step, loss, regCoeff, logFile);
            writer.write("step = " + Double.toString(step) + " score = " + Double.toString(score) + "\n");
            writer.flush();
            writer.close();

        }).start();
    }

//
//    private static void runExperiment(DataLoader.DataFrame data,
//                                      TestGreedyTDSimpleRegion.Optimization<DataLoader.DataFrame, Integer, Double, Class, Double, Double> evaluate,
//                                      int iter, double step,
//                                      Class<?> loss, double regCoeff, String fileName) throws IOException {
//
//        System.out.println(fileName);
//
//        new Thread(() -> {
//
//            File file = new File(fileName);
//            PrintWriter writer;
//            try {
//                writer = new PrintWriter(new FileWriter(file));
//            } catch (IOException e) {
//                System.out.println("Could not create file to write log: " + file);
//                return;
//            }
//            final int numIter = NUM_ITER;
//            DataLoader.DataFrame cur_data;
//            for (int i = 0; i < numIter; i++) {
//                cur_data = bootstrap(data, i);
//                System.out.printf("Iteration No %d\n", i);
//                double score = evaluate.apply(cur_data, iter, step, loss, regCoeff);
//                writer.write(Double.toString(score) + "\n");
//                writer.flush();
//            }
//
//            writer.close();
//
//        }).start();
//    }


    public static void getIntervals() throws IOException {
        String FileName = "intvs_right.txt";
//        DataLoader.TestProcessor processor = new DataLoader.BaseDataReadProcessor();
//        DataLoader.DataFrame data = readData(processor, PATH_TO_DATA, LearnBaseDataName, TestBaseDataName);
//        String path = PATH_TO_DATA + "/base/";
//        if (!Files.exists(Paths.get(path))) {
//            Files.createDirectory(Paths.get(path));
//        }
//
//        runExperimentInterval(data, Utils::findBestRMSEGreedySimpleRegion, 7000, 0.005, L2.class, 0, path + FileName);
//
//        DataLoader.TestProcessor processor = new DataLoader.CTSliceTestProcessor();
//        DataLoader.DataFrame data = readData(processor,
//                PATH_TO_DATA, "slice_train.csv", "slice_test.csv");
//
//        String path = PATH_TO_DATA + "/ct_slice/";
//        if (!Files.exists(Paths.get(path))) {
//            Files.createDirectory(Paths.get(path));
//        }
//
//        runExperimentInterval(data, Utils::findBestRMSEGreedySimpleRegion, 7000, 0.12, L2.class, 0, path + FileName);
//
//        System.out.println("CT SLICE ENDED!");
//
//        String path = PATH_TO_DATA + "/kc_house/";
//        if (!Files.exists(Paths.get(path))) {
//            Files.createDirectory(Paths.get(path));
//        }
//
//        String prefix = path + "intervs_";
//        DataLoader.TestProcessor processor = new DataLoader.KSHouseReadProcessor();
//        DataLoader.DataFrame data = readData(processor, PATH_TO_DATA, "learn_ks_house.csv",
//                "test_ks_house.csv");
//        runExperimentInterval(data, Utils::findBestRMSEGreedyLinearRegion, 7000, 0.02, L2.class, 1e-2, path + FileName, prefix);
//
//        System.out.println("KC HOUSE ENDED!");

        String path = PATH_TO_DATA + "/higgs_linear/";
        if (!Files.exists(Paths.get(path))) {
            Files.createDirectory(Paths.get(path));
        }
        String prefix = path + "intervs_";
        DataLoader.TestProcessor processor = new DataLoader.HIGGSReadProcessor();
        DataLoader.DataFrame data = readData(processor, PATH_TO_DATA, LearnHIGGSFileName, TestHIGGSFileName);
        runExperimentInterval(data, Utils::findBestAUCGreedyLinearRegion, 30, 0.5, L2.class, 0, path + "/" + FileName, prefix);
    }

    public static void getScore() throws IOException {
        String FileName = "res.txt";
//        DataLoader.TestProcessor processor = new DataLoader.BaseDataReadProcessor();
//        DataLoader.DataFrame data = readData(processor, PATH_TO_DATA, LearnBaseDataName, TestBaseDataName);
//        String path = PATH_TO_DATA + "/base/";
//        if (!Files.exists(Paths.get(path))) {
//            Files.createDirectory(Paths.get(path));
//        }
//
//        runExperimentScore(data, Utils::findBestRMSEGreedySimpleRegion, 7000, 0.005, L2.class, 0, path + FileName);
//
//        processor = new DataLoader.CTSliceTestProcessor();
//        data = readData(processor,
//                PATH_TO_DATA, "slice_train.csv", "slice_test.csv");
//
//        path = PATH_TO_DATA + "/ct_slice/";
//        runExperimentScore(data, Utils::findBestRMSEGreedySimpleRegion, 7000, 0.03, L2.class, 0.0, path + FileName);
//
//        System.out.println("CT SLICE ENDED!");
//
        String path = PATH_TO_DATA + "/kc_house/";
        String prefix = path + "res_";

        DataLoader.TestProcessor processor = new DataLoader.KSHouseReadProcessor();
        DataLoader.DataFrame data = readData(processor, PATH_TO_DATA, "learn_ks_house.csv",
                "test_ks_house.csv");

        runExperimentScore(data, Utils::findBestRMSEGreedyLinearRegion, 7000, 0.02, L2.class, 1e-2, path + FileName, prefix);

        System.out.println("KC HOUSE ENDED!");

//        String path = PATH_TO_DATA + "/higgs_linear/";
//        if (!Files.exists(Paths.get(path))) {
//            Files.createDirectory(Paths.get(path));
//        }
//        String prefix = path + "score_";
//        DataLoader.TestProcessor processor = new DataLoader.HIGGSReadProcessor();
//        DataLoader.DataFrame data = readData(processor, PATH_TO_DATA, LearnHIGGSFileName, TestHIGGSFileName);
//        runExperimentScore(data, Utils::findBestAUCGreedyLinearRegion, 12000, 0.5, L2.class, 0.0, path + FileName, prefix);
    }

    public static void testScore() throws IOException {
        String FileName = "test.txt";
//        DataLoader.TestProcessor processor = new DataLoader.BaseDataReadProcessor();
//        DataLoader.DataFrame data = readData(processor, PATH_TO_DATA, LearnBaseDataName, TestBaseDataName);
//        String path = PATH_TO_DATA + "/base/";
//        if (!Files.exists(Paths.get(path))) {
//            Files.createDirectory(Paths.get(path));
//        }
//
//        runTest(data, Utils::findBestRMSEGreedySimpleRegion, 3000, 0.005, L2.class, 0, path + FileName);

        DataLoader.TestProcessor processor = new DataLoader.CTSliceTestProcessor();
        DataLoader.DataFrame data = readData(processor,
                PATH_TO_DATA, "slice_train.csv", "slice_test.csv");

        String path = PATH_TO_DATA + "/ct_slice/";
        runTest(data, Utils::findBestRMSEGreedySimpleRegion, 7000, 0.3, L2.class, 0.0, path + FileName, path);

        System.out.println("CT SLICE ENDED!");
//
//        String path = PATH_TO_DATA + "/kc_house/";
//        DataLoader.TestProcessor processor = new DataLoader.KSHouseReadProcessor();
//        DataLoader.DataFrame data = readData(processor, PATH_TO_DATA, "learn_ks_house.csv",
//                "test_ks_house.csv");
//        runTest(data, Utils::findBestRMSEGreedyLinearRegion, 7000, 0.2, L2.class, 1e-2, path + FileName);
//
//        System.out.println("KC HOUSE ENDED!");

//        String path = PATH_TO_DATA + "/higgs/";
//        if (!Files.exists(Paths.get(path))) {
//            Files.createDirectory(Paths.get(path));
//        }
//
//        DataLoader.TestProcessor processor = new DataLoader.HIGGSReadProcessor();
//        DataLoader.DataFrame data = readData(processor, PATH_TO_DATA, LearnHIGGSFileName,
//                TestHIGGSFileName);
//        runTest(data, Utils::findBestAUCGreedySimpleRegion, 9000, 0.5, L2.class, 0.0, path + FileName);
//    }

//    public static void getResultLinear() throws IOException {
//        String FileNameRes = "res.txt";
//        String FileNameInvs = "invs.txt";
//        DataLoader.TestProcessor processor = new DataLoader.BaseDataReadProcessor();
//        DataLoader.DataFrame data = readData(processor, PATH_TO_DATA, LearnBaseDataName, TestBaseDataName);
//        String path = PATH_TO_DATA + "/base/";
//        if (!Files.exists(Paths.get(path))) {
//            Files.createDirectory(Paths.get(path));
//        }
//
//        runExperimentScore(data, Utils::findBestRMSEGreedyLinearRegion, 7000, 0.005, L2.class, 1e-4, path + FileNameRes);
//        runExperimentInterval(data, Utils::findBestRMSEGreedyLinearRegion, 7000, 0.005, L2.class, 1e-4, path + FileNameInvs);
//
//        processor = new DataLoader.CTSliceTestProcessor();
//        data = readData(processor,
//                PATH_TO_DATA, "slice_train.csv", "slice_test.csv");
//
//        path = PATH_TO_DATA + "/ct_slice/";
//        runExperimentScore(data, Utils::findBestRMSEGreedySimpleRegion, 7000, 0.03, L2.class, 0.0, path + FileName);
//
//        System.out.println("CT SLICE ENDED!");
//
//        path = PATH_TO_DATA + "/kc_house/";
//        processor = new DataLoader.KSHouseReadProcessor();
//        data = readData(processor, PATH_TO_DATA, "learn_ks_house.csv",
//                "test_ks_house.csv");
//        runExperimentScore(data, Utils::findBestRMSEGreedySimpleRegion, 7000, 0.02, L2.class, 0.0, path + FileName);
//
//        System.out.println("KC HOUSE ENDED!");
//
//        path = PATH_TO_DATA + "/higgs/";
//        processor = new DataLoader.HIGGSReadProcessor();
//        data = readData(processor, PATH_TO_DATA, LearnHIGGSFileName, TestHIGGSFileName);
//        runExperimentScore(data, Utils::findBestAUCGreedySimpleRegion, 7000, 0.3, L2.class, 0.0, path + FileName);
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        if (args.length != 1) {
            System.err.println("please provide an argument: path_to_datasets");
            return;
        }

        for (String arg : args) {
            System.out.println(arg);
        }

        PATH_TO_DATA = args[0];

//        getScore();
        getIntervals();
//        testScore();

//        getResultLinear();
    }
}
