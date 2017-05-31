package com.spbsu.ml.loss;

import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.ProgressHandler;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.func.Ensemble;

import java.io.PrintStream;
import java.util.Arrays;

/**
 * igorkuralenok on 23.05.17.
 */
@SuppressWarnings("unused")
public class AUCLogit extends Func.Stub implements TargetFunc {
  protected final Vec target;
  private final DataSet<?> owner;
  private final int allPositive;

  public AUCLogit(final Vec target, final DataSet<?> owner) {
    this.target = target;
    this.owner = owner;
    int positive = 0;
    for (int i = 0; i < target.dim(); i++) {
      if (target.get(i) > 0)
        positive++;
    }
    allPositive = positive;
  }

  private int[] getOrdered(Vec array) {
    int[] order = ArrayTools.sequence(0, array.dim());
    ArrayTools.parallelSort(array.toArray().clone(), order);
    return order;
  }

  @Override
  public double value(Vec x) {
    final double[] weights = new double[x.dim()];
    x.toArray(weights, 0);
    final int[] order = ArrayTools.sequence(0, x.dim());
    ArrayTools.parallelSort(weights, order);
    int trueNegative = 0;
    int falseNegative = 0;

    double sum = 0;
    int curPos = 0;

    double prevFPR = 1;
    double prevTPR = 0;
    double max_accuracy = 0;

    while (curPos < order.length) {
      if (target.get(order[curPos++]) == 1) {
        falseNegative += 1;
        continue;
      }
      else {
        trueNegative += 1;
      }
      final int allNegative = x.dim() - allPositive;
      double falsePositive = allNegative - trueNegative;
      double truePositive = allPositive - falseNegative;
      double TPR = 1.0 * truePositive / allPositive;
      double FPR = 1.0 * falsePositive / allNegative;

//      sum += (TPR + prevTPR)/2 * (prevFPR - FPR);
      sum += TPR * (prevFPR - FPR);
      prevFPR = FPR;
      prevTPR = TPR;

      double cur_accuracy = 1.0 * (trueNegative + truePositive) / (allPositive + allNegative);
      max_accuracy = Math.max(max_accuracy, cur_accuracy);
    }
    return sum;
  }

  public void printResult(Vec x, PrintStream out) {
    final double[] weights = new double[x.dim()];
    x.toArray(weights, 0);
    final int[] order = ArrayTools.sequence(0, x.dim());
    ArrayTools.parallelSort(weights, order);
    int trueNegative = 0;
    int falseNegative = 0;

    double sum = 0;
    double sumT = 0;
    int curPos = 0;

    double prevFPR = 1;
    double prevTPR = 0;
    double max_accuracy = 0;

    while (curPos < order.length) {
      if (target.get(order[curPos++]) == 1) {
        falseNegative += 1;
        continue;
      }
      else {
        trueNegative += 1;
      }
      final int allNegative = x.dim() - allPositive;
      double falsePositive = allNegative - trueNegative;
      double truePositive = allPositive - falseNegative;
      double TPR = 1.0 * truePositive / allPositive;
      double FPR = 1.0 * falsePositive / allNegative;
      out.append(String.valueOf(FPR)).append("\t")
          .append(String.valueOf(TPR)).append("\n");

      sumT += (TPR + prevTPR)/2 * (prevFPR - FPR);
      sum += TPR * (prevFPR - FPR);
      prevFPR = FPR;
      prevTPR = TPR;

      double cur_accuracy = 1.0 * (trueNegative + truePositive) / (allPositive + allNegative);
      max_accuracy = Math.max(max_accuracy, cur_accuracy);
    }
    out.append("AUC Bars: ").append(String.valueOf(sum))
        .append(" AUC Trapezium: ").append(String.valueOf(sumT))
        .append("\n");
  }

  @Override
  public int dim() {
    return target.dim();
  }

  @Override
  public DataSet<?> owner() {
    return owner;
  }
}
