package com.spbsu.ml.loss.multiclass.util;

import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.table.TableBuilder;
import com.spbsu.ml.data.tools.MCTools;

/**
 * User: amosov-f
 * User: qdeee
 * Date: 13.08.14
 * Time: 11:19
 */
public class ConfusionMatrix {
  private final int[][] counts;
  private final int[] fp;
  private final int[] fn;

  public ConfusionMatrix(final int numClasses) {
    counts = new int[numClasses][numClasses];
    fp = new int[numClasses];
    fn = new int[numClasses];
  }

  //TODO[qdeee]: add validation for bad target(same values, missed class)
  public ConfusionMatrix(final IntSeq target, final IntSeq predicted) {
    this(MCTools.countClasses(target));

    for (int i = 0; i < target.length(); i++) {
      final int expected = target.arr[i];
      final int actual = predicted.arr[i];

      if (expected != actual) {
        fn[expected]++;

        if (actual == counts.length)
          //error class label. we should update false negatives and skip others updates
          continue;

        fp[actual]++;
      }
      counts[expected][actual]++;
    }
  }

  public void add(final int expected, final int actual) {
    counts[expected][actual]++;
    if (expected != actual) {
      fn[expected]++;
      if (actual != -1)
        fp[actual]++;
    }
  }

  public double getPrecision(final int c) {
    return (tp(c) + fp(c) > 0) ? tp(c) / (tp(c) + fp(c) + 0.) : 0;
  }

  public double getRecall(final int c) {
    return (tp(c) + fn(c) > 0) ? tp(c) / (tp(c) + fn(c) + 0.) : 0;
  }

  public double getF1Measure(final int clazz) {
    final double p = getPrecision(clazz);
    final double r = getRecall(clazz);
    if (p + r == 0) {
      return 0;
    }
    return 2 * p * r / (p + r);
  }

  public double getMacroPrecision() {
    double macroPrecision = 0;
    for (int i = 0; i < counts.length; i++) {
      macroPrecision += getPrecision(i);
    }
    macroPrecision /= counts.length;
    return macroPrecision;
  }

  public double getMacroRecall() {
    double macroRecall = 0;
    for (int i = 0; i < counts.length; i++) {
      macroRecall += getRecall(i);
    }
    macroRecall /= counts.length;
    return macroRecall;
  }

  public double getMacroF1Measure() {
    final double p = getMacroPrecision();
    final double r = getMacroRecall();
    if (p + r == 0) {
      return 0;
    }
    return 2 * p * r / (p + r);
  }

  public double getMicroPrecision() {
    int tps = 0;
    int fps = 0;
    for (int i = 0; i < counts.length; i++) {
      tps += tp(i);
      fps += fp(i);
    }
    return (tps + fps > 0) ? tps / (tps + fps + 0.) : 0;
  }

  public double getMicroRecall() {
    int tps = 0;
    int fns = 0;
    for (int i = 0; i < counts.length; i++) {
      tps += tp(i);
      fns += fn(i);
    }
    return (tps + fns > 0) ? tps / (tps + fns + 0.) : 0;
  }

  public double getMicroF1Measure() {
    final double p = getMicroPrecision();
    final double r = getMicroRecall();
    if (p + r == 0) {
      return 0;
    }
    return 2 * p * r / (p + r);
  }

  public double getCohenKappa() {
    final int[] sumRows = new int[counts.length];
    final int[] sumColumns = new int[counts.length];
    int sumOfWeights = 0;
    for (int i = 0; i < counts.length; i++) {
      for (int j = 0; j < counts.length; j++) {
        sumRows[i] += counts[i][j];
        sumColumns[j] += counts[i][j];
        sumOfWeights += counts[i][j];
      }
    }
    double correct = 0;
    double chanceAgreement = 0;
    for (int i = 0; i < counts.length; i++) {
      chanceAgreement += (sumRows[i] * sumColumns[i]);
      correct += counts[i][i];
    }
    chanceAgreement /= (sumOfWeights * sumOfWeights);
    correct /= sumOfWeights;

    if (chanceAgreement < 1) {
      return (correct - chanceAgreement) / (1 - chanceAgreement);
    } else {
      return 1;
    }
  }

  public int tp(final int clazz) {
    return counts[clazz][clazz];
  }

  public int fp(final int clazz) {
    return fp[clazz];
  }

  public int fn(final int clazz) {
    return fn[clazz];
  }

  public void add(final ConfusionMatrix confusionMatrix) {
    for (int i = 0; i < counts.length; i++) {
      for (int j = 0; j < counts[i].length; j++) {
        counts[i][j] += confusionMatrix.counts[i][j];
      }
      fp[i] += confusionMatrix.fp[i];
      fn[i] += confusionMatrix.fn[i];
    }
  }

  public int getNumExamples() {
    int count = 0;
    for (int i = 0; i < counts.length; i++) {
      count += ArrayTools.sum(counts[i]);
    }
    return count;
  }

  public int getNumExamples(final int clazz) {
    return ArrayTools.sum(counts[clazz]);
  }

  public int getNumClasses() {
    return counts.length;
  }

  public String toSummaryString() {
    final TableBuilder tableBuilder = new TableBuilder("Metric", "Value");
    final String result = tableBuilder
        .addRow("Micro precision: ",  getMicroPrecision())
        .addRow("Micro recall: ",     getMicroRecall())
        .addRow("Micro F1-measure: ", getMicroF1Measure())
        .addRow("Macro precision: ",  getMacroPrecision())
        .addRow("Macro recall: ",     getMacroRecall())
        .addRow("Macro F1-measure: ", getMacroF1Measure())
        .build();
    return "=== Summary ===\n" + result;
  }

  public String toClassDetailsString() {
    final TableBuilder tableBuilder = new TableBuilder("class", "precision", "recall", "f1-measure");
    for (int i = 0; i < counts.length; i++) {
      tableBuilder.addRow(String.valueOf(i), getPrecision(i), getRecall(i), getF1Measure(i));
    }
    return "=== Detailed Accuracy By Class ===\n" + tableBuilder.build();
  }

  public String oneLineReport() {
    final String f = "%s = %.6f,\t";
    return "{" +
        String.format(f, "mP", getMicroPrecision()) +
        String.format(f, "mR", getMicroRecall()) +
        String.format(f, "mF", getMicroF1Measure()) +
        String.format(f, "MP", getMacroPrecision()) +
        String.format(f, "MR", getMacroRecall()) +
        String.format(f, "MF", getMacroF1Measure()) + "}";
  }

  public String debug() {
    return String.format("%.6f|%.6f|%.6f|%.6f|%.6f|%.6f|",
        getMicroPrecision(), getMicroRecall(), getMicroF1Measure(), getMacroPrecision(), getMacroRecall(), getMacroF1Measure());
  }

  @Override
  public String toString() {
    final StringBuilder sb = new StringBuilder("=== Confusion Matrix ===\n");
    for (int i = 0; i < counts.length; i++) {
      for (int j = 0; j < counts.length; j++) {
        sb.append(String.format("%6d\t", counts[i][j]));
      }
      sb.append("\n");
    }
    return sb.toString();
  }
}