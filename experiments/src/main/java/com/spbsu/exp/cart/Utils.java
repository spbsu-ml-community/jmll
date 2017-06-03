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
import com.spbsu.ml.methods.BootstrapOptimization;
import com.spbsu.ml.methods.GradientBoosting;

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

    AUCCalcer(final String message, final VecDataSet ds, final Vec rightAns) {
      this(message, ds, rightAns, true);
    }

    AUCCalcer(final String message, final VecDataSet ds, final Vec rightAns, boolean isWrite) {
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
      if (isWrite) System.out.print(message + value);
      max = Math.max(value, max);
      if (isWrite) System.out.print(" best = " + max);

      System.out.printf(" rate = %.5f", max_accuracy);
    }
  }

  static double findBestAUC(final DataLoader.DataFrame data, final int iterations, final double step,
                                    final Class func, final double regCoeff) {
    final GradientBoosting<LLLogit> boosting = new GradientBoosting<>(
            new BootstrapOptimization<>(
                    new com.spbsu.exp.cart.CARTTreeOptimization(
                            GridTools.medianGrid(data.getLearnFeatures(), 32), 6, regCoeff), RND), func, iterations, step);
    final Action counter = new ProgressHandler() {
      int index = 0;

      @Override
      public void invoke(final Trans partial) {
        System.out.print("\n" + index++);
      }
    };
    final LLLogit learnTarget = new LLLogit(data.getLearnTarget(), data.getLearnFeatures());
    final LLLogit testTarget = new LLLogit(data.getTestTarget(), data.getTestFeatures());

    final AUCCalcer aucCalcerLearn = new AUCCalcer("\tAUC learn:\t", data.getLearnFeatures(),
            data.getLearnTarget());
    final AUCCalcer aucCalcerTest = new AUCCalcer("\tAUC test:\t", data.getTestFeatures(),
            data.getTestTarget());

    boosting.addListener(counter);
    boosting.addListener(aucCalcerLearn);
    boosting.addListener(aucCalcerTest);
    boosting.fit(data.getLearnFeatures(), learnTarget);
    return aucCalcerTest.getMax();
  }

  static double findBestRMSE(final DataLoader.DataFrame data, final int iterations, final double step,
                                     final Class funcClass, final double regCoeff) {
    final GradientBoosting<L2> boosting = new GradientBoosting<>(
            new BootstrapOptimization<>(
                    new com.spbsu.exp.cart.CARTTreeOptimization(
                            GridTools.medianGrid(data.getLearnFeatures(), 32), 6, regCoeff), RND), funcClass, iterations, step);
    final Action counter = new ProgressHandler() {
      int index = 0;

      @Override
      public void invoke(final Trans partial) {
        System.out.print("\n" + index++);
      }
    };
    final L2 learnTarget = new L2(data.getLearnTarget(), data.getLearnFeatures());
    final L2 testTarget = new L2(data.getTestTarget(), data.getTestFeatures());
    final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", data.getLearnFeatures(), learnTarget);
    final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", data.getTestFeatures(), testTarget);

//    boosting.addListener(counter);
    boosting.addListener(learnListener);
    boosting.addListener(validateListener);
    boosting.fit(data.getLearnFeatures(), learnTarget);
    return validateListener.getMinRMSE();
  }

  protected static class ScoreCalcer implements ProgressHandler {
    private final String message;
    private final Vec current;
    private final VecDataSet ds;
    private final TargetFunc target;
    private boolean isWrite = true;

    ScoreCalcer(final String message, final VecDataSet ds, final TargetFunc target) {
      this(message, ds, target, true);
    }

    ScoreCalcer(final String message, final VecDataSet ds, final TargetFunc target, boolean isWrite) {
      this.message = message;
      this.isWrite = isWrite;
      this.ds = ds;
      this.target = target;
      current = new ArrayVec(ds.length());
    }

    double min = 1e10;

    double getMinRMSE() {
      return min;
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

//      if (isWrite) System.out.print(message + " " + value);
      min = Math.min(value, min);
//      if (isWrite) System.out.print(" best = " + min);
    }
  }
}
