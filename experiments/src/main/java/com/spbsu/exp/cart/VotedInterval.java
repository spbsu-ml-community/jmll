package com.spbsu.exp.cart;

import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.LOOL2;

import java.io.*;

import static com.spbsu.exp.cart.DataLoader.bootstrap;
import static com.spbsu.exp.cart.DataLoader.readData;

public class VotedInterval {
  private static String PATH_TO_DATA;
  private static final Integer NUM_ITER = 100;
  private static String PYTHON_PATH;

  @FunctionalInterface
  interface Optimization<Data, NumIter, Step, Loss, RegCoeff, Score> {
    Score apply(Data data, NumIter numIter, Step step, Loss loss, RegCoeff e);
  }

  private static void runXGBoost(DataLoader.DataFrame data, String folder) throws IOException, InterruptedException {
    DataLoader.DataFrame cur_data;
    for (int i = 0; i < NUM_ITER; i++) {
      cur_data = bootstrap(data, i);
      System.out.printf("Iteration No %d\n", i);
      try {
        cur_data.toFile(folder);
      } catch (FileNotFoundException e) {
        System.err.println("Folder with python scripts not found");
        return;
      }
      String[] cmd = {PYTHON_PATH, folder + "/boost.py"};

      Process xgboost;
      try {
        xgboost = Runtime.getRuntime().exec(cmd);
      } catch (IOException e) {
        System.err.println("Couldn't start python process.");
        return;
      }
      InputStream outStream = xgboost.getInputStream();
      InputStream errorStream = xgboost.getErrorStream();

      BufferedReader out = new BufferedReader(new InputStreamReader(outStream));
      BufferedReader err = new BufferedReader(new InputStreamReader(errorStream));

      while (xgboost.isAlive()) {
        out.lines().forEach(System.out::println);
        err.lines().forEach(System.err::println);
      }
    }
  }

  private static void runExperiment(DataLoader.DataFrame data,
                                    Optimization<DataLoader.DataFrame, Integer, Double, Class, Double, Double> evaluate,
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
      }

      writer.close();

    }).start();
  }

  public static void main(String[] args) throws IOException, InterruptedException {
    if (args.length != 2) {
      System.err.println("please provide two arguments: path_to_datasets full_path_to_python");
      return;
    }

    for (String arg : args) {
      System.out.println(arg);
    }

    PATH_TO_DATA = args[0];
    PYTHON_PATH = args[1];

    DataLoader.TestProcessor processor = new DataLoader.CTSliceTestProcessor();
    DataLoader.DataFrame data = readData(processor,
            PATH_TO_DATA, "slice_train.csv", "slice_test.csv");

//    String path = PATH_TO_DATA + "/ct_slice/";
//    runExperiment(data, Utils::findBestRMSE, 3300, 0.03, L2.class, 0.0, path + "/without.txt");
//    runExperiment(data, Utils::findBestRMSE, 3300, 0.03, LOOL2.class, 0.0, path + "/loo.txt");
//    runExperiment(data, Utils::findBestRMSE, 3300, 0.03, CARTSteinEasy.class, 0.0, path + "/bayes.txt");
//    runExperiment(data, Utils::findBestRMSE, 3300, 0.03, L2.class, 0.4, path + "/reg.txt");
//    runExperiment(data, Utils::findBestRMSE, 3300, 0.03, CARTSteinEasy.class, 0.4, path + "/combo.txt");
//    runXGBoost(data, path);
//
//    System.out.println("CT SLICE ENDED!");
//
//    path = PATH_TO_DATA + "/kc_house/";
//    processor = new DataLoader.KSHouseReadProcessor();
//    data = readData(processor, PATH_TO_DATA, "learn_ks_house.csv",
//            "test_ks_house.csv");
//    runExperiment(data, Utils::findBestRMSE, 2600, 0.008, L2.class, 0.0, path + "/without.txt");
//    runExperiment(data, Utils::findBestRMSE, 2600, 0.008, LOOL2.class, 0.0, path + "/loo.txt");
//    runExperiment(data, Utils::findBestRMSE, 2600, 0.008, CARTSteinEasy.class, 0.0, path + "/bayes.txt");
//    runExperiment(data, Utils::findBestRMSE, 2600, 0.008, L2.class, 0.4, path + "/reg.txt");
//    runExperiment(data, Utils::findBestRMSE, 2600, 0.008, CARTSteinEasy.class, 0.4, path + "/combo.txt");
//    runXGBoost(data, path);
//
//    System.out.println("KC HOUSE ENDED!");
//
//    path = PATH_TO_DATA + "/higgs/";
//    processor = new DataLoader.HIGGSReadProcessor();
//    data = readData(processor, PATH_TO_DATA, "HIGGS_learn_1M.csv",
//            "HIGGS_test.csv");
//    runExperiment(data, Utils::findBestAUC, 4000, 0.3, L2.class, 0.0, path + "/without.txt");
//    runExperiment(data, Utils::findBestAUC, 4000, 0.3, LOOL2.class, 0.0, path + "/loo.txt");
//    runExperiment(data, Utils::findBestAUC, 4000, 0.3, CARTSteinEasy.class, 0.0, path + "/bayes.txt");
//    runExperiment(data, Utils::findBestAUC, 4000, 0.3, L2.class, 0.4, path + "/reg.txt");
//    runExperiment(data, Utils::findBestAUC, 4000, 0.3, CARTSteinEasy.class, 0.4, path + "/combo.txt");
//    runXGBoost(data, path);
  }
}
