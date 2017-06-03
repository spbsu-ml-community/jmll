package com.spbsu.exp.cart;

import com.spbsu.ml.loss.*;

import java.io.*;
import java.util.Arrays;

import static com.spbsu.exp.cart.DataLoader.bootstrap;
import static com.spbsu.exp.cart.DataLoader.readData;
import static com.spbsu.exp.cart.Utils.findBestAUC;
import static com.spbsu.exp.cart.Utils.findBestRMSE;


/**
 * Created by n_buga on 23.02.17.
 * Estimate the accuracy of the CART tree.
 */
public class ConfidentIntervalFinder {
  private static final String dir = "ml/src/test/resources/com/spbsu/ml";
  //    private static final String dir = "./resources";
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
  private static final String ResultFile = "result2.txt";

  public static void main(String[] args) throws IOException {

    try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
            new FileOutputStream("result2.txt"), "utf-8"))) {
//            findIntervalCancerData("L2", L2.class, writer, 0);
//            findIntervalCancerData("SteinDifficult", CARTSteinDifficult.class, writer, 0);
//            findIntervalCancerData("SteinEasy", CARTSteinEasy.class, writer, 0);
//            findIntervalCancerData("LOO", LOOL2.class, writer, 0);
//            findIntervalCancerData("SAT", SatL2.class, writer, 0);
//            findIntervalCancerData("Reg with 0.4", L2.class, writer, 0.4);
//            findIntervalCancerData("SAT + SteinDifficult", CARTSatSteinL2.class, writer, 0);


//            findIntervalCTSliceData("L2", L2.class, writer, 0, 0.03, 3300, 1);
//            findIntervalCTSliceData("LOO", LOOL2.class, writer, 0, 0.03, 5300, 1);
//            findIntervalCTSliceData("SAT", SatL2.class, writer, 0, 0.03, 3300, 1);
//            findIntervalCTSliceData("SAT + SteinDifficult", CARTSatSteinL2.class, writer, 0, 0.03, 3300, 1);
//            findIntervalCTSliceData("CARTSteinDifficult", JNL2.class, writer, 0, 0.03, 5300, 1);
//            findIntervalCTSliceData("CARTSteinEasy", CARTSteinEasy.class, writer, 0, 0.03, 3300, 1);
//            findIntervalCTSliceData("Reg with 2", L2Reg.class, writer, 0, 0.025, 3300, 1);
//            findIntervalCTSliceData("Reg + LOO + Stein--", CARTSteinEasyReg.class, writer, 0, 0.03, 3300, 1);

//            findIntervalHiggsData("L2", L2.class, writer, 0);
//            findIntervalHiggsData("LOO", LOOL2.class, writer, 0);
//            findIntervalHiggsData("SAT", SatL2.class, writer, 0);
//            findIntervalHiggsData("CARTSteinDifficult", CARTSteinDifficult.class, writer, 0);
//            findIntervalHiggsData("CARTSteinEasy", CARTSteinEasy.class, writer, 0);
//            findIntervalHiggsData("Reg with 0.4", CARTL2.class, writer, 0.4);
//
      findIntervalKSHouseData("L2", L2.class, writer, 0);
//            findIntervalKSHouseData("LOO", LOOL2.class, writer, 0);
//            findIntervalKSHouseData("SAT", SatL2.class, writer, 0);
//            findIntervalKSHouseData("CARTSteinDifficult", CARTSteinDifficult.class, writer, 0);
//            findIntervalKSHouseData("CARTSteinEasy", CARTSteinEasy.class, writer, 0);
//            findIntervalKSHouseData("Reg with 0.4", CARTL2.class, writer, 0.4);
    }
    System.exit(0);
  }

  private static void findIntervalCancerData(String msg, Class funcClass, BufferedWriter writer, double regCoeff) {
    int M = 100;
    int iterations = 1000;
    double step = 0.05;
    int depth = 3;

    double best[] = new double[M];

    try {
      DataLoader.TestProcessor processor = new DataLoader.CancerTestProcessor();

      DataLoader.DataFrame data = readData(processor, dir, LearnCancerFileName, TestCancerFileName);

      DataLoader.DataFrame cur_data = data;
      for (int i = 0; i < M; i++) {
        System.out.printf("!!!%d", i);
        double auc = findBestAUC(cur_data, iterations, step, funcClass, regCoeff);
        best[i] = auc;
        System.out.printf("\nThe Best AUC for cancerData = %.4fc\n", auc);
        cur_data = bootstrap(data, System.currentTimeMillis());
      }
    } catch (IOException e) {
      e.printStackTrace();
    } finally {
      Arrays.sort(best);
      int i = 0;
      while (i < M && best[i] == 0) {
        i++;
      }
      String info = String.format("The interval for cancerData + %s: %d times, %d iterations, %.4f step," +
                      "%d depth [%.7f, %.7f]\n",
              msg, M, iterations, step, depth,
              best[i + 5], best[M - 6]);
//            String info = String.format("The value for %d iterations and %.4f step is %.7f",
//                    iterations, step, best[0]);
      System.out.printf(info);
      try {
        writer.write(info);
        writer.flush();
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
  }

  private static void findIntervalKSHouseData(String msg, Class funcClass, BufferedWriter writer, double regCoeff) {
    int M = 100;
    int iterations = 200;
    double step = 0.1;

    double best[] = new double[M];

    try {
      DataLoader.TestProcessor processor = new DataLoader.KSHouseReadProcessor();

      DataLoader.DataFrame data = readData(processor, dir, LearnKSHouseFileName, TestKSHouseFileName);

      DataLoader.DataFrame cur_data = data;
      for (int i = 0; i < M; i++) {
        System.out.printf("!!!%d", i);
        double rmse = findBestRMSE(cur_data, iterations, step, funcClass, regCoeff);
        best[i] = rmse;
        System.out.printf("\nThe Best RMSE for ks_House = %.4fc\n", rmse);
        cur_data = bootstrap(data, System.currentTimeMillis());
      }
    } catch (IOException | NumberFormatException e) {
      e.printStackTrace();
    } finally {
      Arrays.sort(best);
      int i = 0;
      while (i < M && best[i] == 0) {
        i++;
      }
      String info = String.format("The interval for ks_house + %s: %d times, %d iterations, %.4f step, [%.7f, %.7f]\n",
              msg, M, iterations, step,
              best[i + 5], best[M - 6]);
      System.out.printf(info);
      try {
        writer.write(info);
        writer.flush();
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
  }

  private static void findIntervalHiggsData(String msg, Class funcClass, BufferedWriter writer, double regCoeff) {
    int M = 100;
    int iterations = 150;
    double step = 0.1;

    double best[] = new double[M];

    try {
      DataLoader.TestProcessor processor = new DataLoader.HIGGSReadProcessor();

      DataLoader.DataFrame data = readData(processor, dir, LearnHIGGSFileName, TestHIGGSFileName);

      DataLoader.DataFrame cur_data = data;
      for (int i = 0; i < M; i++) {
        System.out.printf("!!!%d", i);
        double auc = findBestAUC(cur_data, iterations, step, funcClass, regCoeff);
        best[i] = auc;
        System.out.printf("\nThe Best AUC HIGGS = %.4fc\n", auc);
        cur_data = bootstrap(data, System.currentTimeMillis());
      }
    } catch (IOException e) {
      e.printStackTrace();
    } finally {
      Arrays.sort(best);
      int i = 0;
      while (i < M && best[i] == 0) {
        i++;
      }
      String info = String.format("The interval for HIGGS + %s: %d times, %d iterations, %.4f step, [%.7f, %.7f]\n",
              msg, M, iterations, step,
              best[i + 5], best[M - 6]);
      System.out.printf(info);
      try {
        writer.write(info);
        writer.flush();
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
  }

  private static void findIntervalCTSliceData(String msg, Class funcClass, BufferedWriter writer,
                                              double regCoeff, double step,
                                              int iterations, int averageSize) {
    double best[] = new double[averageSize];

    try {
      DataLoader.CTSliceTestProcessor processor = new DataLoader.CTSliceTestProcessor();
      DataLoader.DataFrame data = readData(processor, dir, LearnCTSliceFileName, TestCTSliceFileName);
      DataLoader.DataFrame cur_data = data;
      for (int i = 0; i < averageSize; i++) {
        double rmse = findBestRMSE(cur_data, iterations, step, funcClass, regCoeff);
        best[i] = rmse;
        System.out.printf("\nThe Best RMSE for CTSlices = %.4fc\n", rmse);
        cur_data = bootstrap(data, System.currentTimeMillis());
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
