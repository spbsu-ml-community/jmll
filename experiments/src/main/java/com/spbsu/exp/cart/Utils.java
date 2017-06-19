package com.spbsu.exp.cart;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.GridTools;
import com.spbsu.ml.ProgressHandler;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.BootstrapOptimization;
import com.spbsu.ml.methods.GradientBoosting;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.greedyRegion.GreedyTDLinearRegion;
import com.spbsu.ml.methods.greedyRegion.GreedyTDProbRegion;
import com.spbsu.ml.methods.greedyRegion.GreedyTDSimpleRegion;

import javax.xml.crypto.Data;
import java.io.*;

public class Utils {
  private static final FastRandom RND = new FastRandom(System.currentTimeMillis());

  private static class AUCCalcer implements ProgressHandler {
    private final String message;
    private final Vec current;
    private final VecDataSet ds;
    private final Vec rightAns;
    private int allNegative = 0;
    private int allPositive = 0;
    private boolean isWrite = true;
    private PrintWriter writer;

    AUCCalcer(final String message, final VecDataSet ds, final Vec rightAns, PrintWriter writer) {
      this(message, ds, rightAns, true, writer);
    }

    AUCCalcer(final String message, final VecDataSet ds, final Vec rightAns, boolean isWrite, PrintWriter writer) {
      this.writer = writer;
      this.message = message;
      this.ds = ds;
      this.isWrite = isWrite;
      this.rightAns = rightAns;
      current = new ArrayVec(ds.length());
      for (int i = 0; i < rightAns.dim(); i++) {
        if (rightAns.at(i) == 1) {
          allPositive += 1;
        } else {
          allNegative += 1;
        }
      }
    }

    private double max = 0;

    double getMax() {
      return max;
    }

    private int[] getOrdered(Vec array) {
      int[] order = ArrayTools.sequence(0, array.dim());
      ArrayTools.parallelSort(array.toArray().clone(), order);
      return order;
    }

    @Override
    public void invoke(final Trans partial) {
      int length = ds.length();

      if (partial instanceof Ensemble) {
        final Ensemble linear = (Ensemble) partial;
        final Trans increment = linear.last();
        for (int i = 0; i < length; i++) {
          if (increment instanceof Ensemble) {
            current.adjust(i, linear.wlast() * (increment.trans(ds.data().row(i)).get(0)));
          } else {
            current.adjust(i, linear.wlast() * ((Func) increment).value(ds.data().row(i)));
          }
        }
      } else {
        for (int i = 0; i < length; i++) {
          current.set(i, ((Func) partial).value(ds.data().row(i)));
        }
      }

      int ordered[] = getOrdered(current);
      int trueNegative = 0;
      int falseNegative = 0;

      double sum = 0;
      int curPos = 0;

      double prevFPR = 1;

/*            ArrayList<Double> x = new ArrayList<>();
            ArrayList<Double> y = new ArrayList<>(); */

      double max_accuracy = 0;

      while (curPos < ordered.length) {
        if (rightAns.get(ordered[curPos++]) != 1) { //!!!!
          trueNegative += 1;
        } else {
          falseNegative += 1;
          continue;
        }
        double falsePositive = allNegative - trueNegative;
        double truePositive = allPositive - falseNegative;
        double TPR = 1.0 * truePositive / allPositive;
        double FPR = 1.0 * falsePositive / allNegative;

//                x.add(FPR);
//                y.add(TPR);
        sum += TPR * (prevFPR - FPR);
        prevFPR = FPR;

        double cur_accuracy = 1.0 * (trueNegative + truePositive) / (allPositive + allNegative);
        max_accuracy = Math.max(max_accuracy, cur_accuracy);
      }

/*            XYChart ex = org.knowm.xchart.QuickChart.getChart("Simple chart", "x", "y",
                    "y(x)", x, y);
            new org.knowm.xchart.SwingWrapper<>(ex).displayChart(); */

      final double value = sum;
      if (isWrite) {
        writer.write(message + Double.toString(value));
        writer.flush();
      }
      max = Math.max(value, max);
      if (isWrite) {
        writer.write(" best = " + max);
        writer.flush();
      }

      writer.write(String.format(" rate = %.5f", max_accuracy));
    }
  }

  static double findBestAUC(final DataLoader.DataFrame data, final int iterations, final double step,
                                    final Class func, final double regCoeff, String logFile) {
    File file = new File(logFile);
    PrintWriter writer;
    try {
      writer = new PrintWriter(new FileWriter(file));
    } catch (IOException e) {
      System.out.println("Could not create logFile: " + file);
      return 0;
    }

    final GradientBoosting<LLLogit> boosting = new GradientBoosting<>(
            new BootstrapOptimization<>(
                    new com.spbsu.ml.methods.cart.CARTTreeOptimization<>(
                            GridTools.medianGrid(data.getLearnFeatures(), 32), 6, regCoeff), RND), func, iterations, step);
    final Action counter = new ProgressHandler() {
      int index = 0;

      @Override
      public void invoke(final Trans partial) {
        writer.write("\n" + index++);
      }
    };
    final LLLogit learnTarget = new LLLogit(data.getLearnTarget(), data.getLearnFeatures());
    final LLLogit testTarget = new LLLogit(data.getTestTarget(), data.getTestFeatures());

    final AUCCalcer aucCalcerLearn = new AUCCalcer("\tAUC learn:\t", data.getLearnFeatures(),
            data.getLearnTarget(), writer);
    final AUCCalcer aucCalcerTest = new AUCCalcer("\tAUC test:\t", data.getTestFeatures(),
            data.getTestTarget(), writer);

    boosting.addListener(counter);
    boosting.addListener(aucCalcerLearn);
    boosting.addListener(aucCalcerTest);
    boosting.fit(data.getLearnFeatures(), learnTarget);
    return aucCalcerTest.getMax();
  }

  static double findBestRMSE(final DataLoader.DataFrame data, final int iterations, final double step,
                                     final Class funcClass, final double regCoeff, String logFile) {

    File file = new File(logFile);
    PrintWriter writer;
    try {
      writer = new PrintWriter(new FileWriter(file));
    } catch (IOException e) {
      System.out.println("Could not create logFile: " + file);
      return 0;
    }

    final GradientBoosting<L2> boosting = new GradientBoosting<>(
            new BootstrapOptimization<>(
                    new com.spbsu.ml.methods.cart.CARTTreeOptimization(
                            GridTools.medianGrid(data.getLearnFeatures(), 32), 6, regCoeff), RND), funcClass, iterations, step);
    final Action counter = new ProgressHandler() {
      int index = 0;

      @Override
      public void invoke(final Trans partial) {
        writer.write("\n" + index);
        index++;
        writer.flush();
      }
    };
    final L2 learnTarget = new L2(data.getLearnTarget(), data.getLearnFeatures());
    final L2 testTarget = new L2(data.getTestTarget(), data.getTestFeatures());
    final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", data.getLearnFeatures(), learnTarget, writer);
    final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", data.getTestFeatures(), testTarget, writer);

    boosting.addListener(counter);
    boosting.addListener(learnListener);
    boosting.addListener(validateListener);
    boosting.fit(data.getLearnFeatures(), learnTarget);
    return validateListener.getMinRMSE();
  }

  static double findBestAUCRegions(final VecOptimization.Stub<WeightedLoss<? extends L2>> weak,
                                   final DataLoader.DataFrame data, final int iterations, final double step,
                                   final Class funcClass, String logFile) {

    File file = new File(logFile);
    PrintWriter writer;
    try {
      writer = new PrintWriter(new FileWriter(file));
    } catch (IOException e) {
      System.out.println("Could not create logFile: " + file);
      return 0;
    }


    final GradientBoosting<LLLogit> boosting = new GradientBoosting<>(
            new BootstrapOptimization<L2>(weak, RND), funcClass, iterations, step);

    final Action counter = new ProgressHandler() {
      int index = 0;

      @Override
      public void invoke(final Trans partial) {
        writer.write("\n" + Integer.toString(index) + "\n");
        writer.flush();
        index++;
      }
    };
    final LLLogit learnTarget = new LLLogit(data.getLearnTarget(), data.getLearnFeatures());
    final LLLogit testTarget = new LLLogit(data.getTestTarget(), data.getTestFeatures());

    final AUCCalcer aucCalcerLearn = new AUCCalcer("\tAUC learn:\t", data.getLearnFeatures(),
            data.getLearnTarget(), writer);
    final AUCCalcer aucCalcerTest = new AUCCalcer("\tAUC test:\t", data.getTestFeatures(),
            data.getTestTarget(), writer);

    boosting.addListener(counter);
    boosting.addListener(aucCalcerLearn);
    boosting.addListener(aucCalcerTest);
    boosting.fit(data.getLearnFeatures(), learnTarget);
    writer.close();
    return aucCalcerTest.getMax();
  }

  static double findBestRMSERegions(final VecOptimization.Stub<WeightedLoss<? extends L2>> weak,
                                    final DataLoader.DataFrame data, final int iterations, final double step,
                                    final Class funcClass, String logFile) {
    File file = new File(logFile);
    PrintWriter writer;
    try {
      writer = new PrintWriter(new FileWriter(file));
    } catch (IOException e) {
      System.out.println("Could not create logFile: " + file);
      return 0;
    }


    final GradientBoosting<L2> boosting = new GradientBoosting<>(
            new BootstrapOptimization<L2>(weak, RND), funcClass, iterations, step);
    final Action counter = new ProgressHandler() {
      int index = 0;

      @Override
      public void invoke(final Trans partial) {
        writer.write("\n" + index++);
      }
    };
    final L2 learnTarget = new L2(data.getLearnTarget(), data.getLearnFeatures());
    final L2 testTarget = new L2(data.getTestTarget(), data.getTestFeatures());
    final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", data.getLearnFeatures(), learnTarget, writer);
    final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", data.getTestFeatures(), testTarget, writer);

    boosting.addListener(counter);
    boosting.addListener(learnListener);
    boosting.addListener(validateListener);
    boosting.fit(data.getLearnFeatures(), learnTarget);
    return validateListener.getMinRMSE();
  }

  public static double findBestRMSEGreedyProbRegion(final DataLoader.DataFrame data, final int iterations, final double step,
                                              final Class funcClass, final double regCoeff, double beta, double alpha, String logFile) {
    return findBestRMSERegions(new GreedyTDProbRegion<>(
                    GridTools.medianGrid(data.getLearnFeatures(), 32), 7, regCoeff, beta, alpha), data,
    iterations, step, funcClass,logFile);
  }

  public static double findBestRMSEGreedyLinearRegion(final DataLoader.DataFrame data, final int iterations, final double step,
                                                      final Class funcClass, double regGoeff, String logFile) {
    return findBestRMSERegions(new GreedyTDLinearRegion<>(
                    GridTools.medianGrid(data.getLearnFeatures(), 32), 6, regGoeff), data,
            iterations, step, funcClass, logFile);
  }

  public static double findBestAUCGreedyLinearRegion(final DataLoader.DataFrame data, final int iterations, final double step,
                                                     final Class funcClass, double regGoeff, String logFile) {
    return findBestAUCRegions(new GreedyTDLinearRegion<>(
                    GridTools.medianGrid(data.getLearnFeatures(), 32), 15, regGoeff), data,
            iterations, step, funcClass, logFile);
  }

  public static double findBestRMSEGreedySimpleRegion(final DataLoader.DataFrame data, final int iterations, final double step,
                                                      final Class funcClass, final double regCoeff, String logFile) {
    return findBestRMSERegions(new GreedyTDSimpleRegion<>(
                    GridTools.medianGrid(data.getLearnFeatures(), 32), 15, regCoeff), data,
            iterations, step, funcClass, logFile);
  }

  public static double findBestAUCGreedySimpleRegion(final DataLoader.DataFrame data, final int iterations, final double step,
                                                     final Class funcClass, final double regCoeff, String logFile) {
    return findBestAUCRegions(new GreedyTDSimpleRegion<>(GridTools.medianGrid(data.getLearnFeatures(), 32), 7, regCoeff), data,
            iterations, step, funcClass, logFile);
  }

  protected static class ScoreCalcer implements ProgressHandler {
    private final String message;
    private final Vec current;
    private final VecDataSet ds;
    private final TargetFunc target;
    private boolean isWrite = true;
    private PrintWriter writer;

    ScoreCalcer(final String message, final VecDataSet ds, final TargetFunc target, PrintWriter writer) {
      this(message, ds, target, true, writer);
    }

    ScoreCalcer(final String message, final VecDataSet ds, final TargetFunc target, boolean isWrite, PrintWriter writer) {
      this.writer = writer;
      this.message = message;
      this.isWrite = isWrite;
      this.ds = ds;
      this.target = target;
      current = new ArrayVec(ds.length());
    }

    double mmin = 1e10;

    double getMinRMSE() {
      return mmin;
    }

    @Override
    public void invoke(final Trans partial) {
      if (partial instanceof Ensemble) {
        final Ensemble linear = (Ensemble) partial;
        final Trans increment = linear.last();
        for (int i = 0; i < ds.length(); i++) {
          if (increment instanceof Ensemble) {
            current.adjust(i, linear.wlast() * (increment.trans(ds.data().row(i)).get(0)));
          } else {
            current.adjust(i, linear.wlast() * ((Func) increment).value(ds.data().row(i)));
          }
        }
      } else {
        for (int i = 0; i < ds.length(); i++) {
          current.set(i, ((Func) partial).value(ds.data().row(i)));
        }
      }

      final double value = target.value(current);

      if (isWrite) {
        writer.write(message + " " + value);
        writer.flush();
      }
      mmin = Math.min(value, mmin);
      if (isWrite) {
        writer.write(" best = " + mmin);
        writer.flush();
      }
      if (message.equals("\ttest:\t"))
        writer.write("\n");
        writer.flush();
    }
  }
}
