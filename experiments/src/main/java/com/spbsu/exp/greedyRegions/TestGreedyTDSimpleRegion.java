package com.spbsu.exp.greedyRegions;

import com.spbsu.exp.cart.CARTSteinEasy;
import com.spbsu.exp.cart.DataLoader;
import com.spbsu.exp.cart.Utils;
import com.spbsu.exp.cart.VotedInterval;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.LOOL2;
import javafx.scene.shape.Path;

import javax.xml.crypto.Data;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;

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
    private static final Integer NUM_ITER = 1;

    @FunctionalInterface
    interface Optimization<Data, NumIter, Step, Loss, RegCoeff, Score> {
        Score apply(Data data, NumIter numIter, Step step, Loss loss, RegCoeff e);
    }

    private static void runExperiment(DataLoader.DataFrame data,
                                      TestGreedyTDSimpleRegion.Optimization<DataLoader.DataFrame, Integer, Double, Class, Double, Double> evaluate,
                                      int iter, double step,
                                      Class<?> loss, double regCoeff, String fileName) throws IOException {

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
                System.out.printf("Iteration No %d\n", i);
                double score = evaluate.apply(cur_data, iter, step, loss, regCoeff);
                writer.write(Double.toString(score) + "\n");
                writer.flush();
            }

            writer.close();

        }).start();
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

        DataLoader.TestProcessor processor = new DataLoader.BaseDataReadProcessor();
        DataLoader.DataFrame data = readData(processor, PATH_TO_DATA, LearnBaseDataName, TestBaseDataName);
        String path = PATH_TO_DATA + "/base/";
//        if (!Files.exists(Paths.get(path))) {
//            Files.createDirectory(Paths.get(path));
//        }
//
//        runExperiment(data, Utils::findBestRMSEGreedySimpleRegion, 3000, 0.03, L2.class, 1e-3, path + "/res.txt");
//
//        processor = new DataLoader.CTSliceTestProcessor();
//        data = readData(processor,
//                PATH_TO_DATA, "slice_train.csv", "slice_test.csv");
//
//        path = PATH_TO_DATA + "/ct_slice/";
//        if (!Files.exists(Paths.get(path))) {
//            Files.createDirectory(Paths.get(path));
//        }
//
//        runExperiment(data, Utils::findBestRMSEGreedySimpleRegion, 3000, 0.005, L2.class, 1e-3, path + "/res.txt");
//
//        System.out.println("CT SLICE ENDED!");
//
        path = PATH_TO_DATA + "/kc_house/";
        if (!Files.exists(Paths.get(path))) {
            Files.createDirectory(Paths.get(path));
        }

        processor = new DataLoader.KSHouseReadProcessor();
        data = readData(processor, PATH_TO_DATA, "learn_ks_house.csv",
                "test_ks_house.csv");
        runExperiment(data, Utils::findBestRMSEGreedySimpleRegion, 7000, 0.008, L2.class, 0, path + "/res.txt");
//
//        System.out.println("KC HOUSE ENDED!");

//        path = PATH_TO_DATA + "/higgs/";
//        if (!Files.exists(Paths.get(path))) {
//            Files.createDirectory(Paths.get(path));
//        }
//        processor = new DataLoader.HIGGSReadProcessor();
//        data = readData(processor, PATH_TO_DATA, "HIGGS_learn_1M.csv",
//                "HIGGS_test.csv");
//        runExperiment(data, Utils::findBestAUCGreedySimpleRegion, 3000, 0.003, L2.class, 1e-3, path + "/res.txt");
    }
}
