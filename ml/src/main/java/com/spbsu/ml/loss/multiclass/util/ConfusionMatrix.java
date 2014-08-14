package com.spbsu.ml.loss.multiclass.util;

/**
 * User: amosov-f
 * Date: 13.08.14
 * Time: 11:19
 */
public class ConfusionMatrix {

  private final int[][] counts;

  public ConfusionMatrix(int numClasses) {
    counts = new int[numClasses][numClasses];
  }

  public void add(int expected, int actual) {
    counts[expected][actual]++;
  }

  public double getPrecision(int clazz) {
    int tp = getTruePositives(clazz);
    int fp = getFalsePositives(clazz);
    if (tp + fp == 0) {
      return 0;
    }
    return (double) tp / (tp + fp);
  }

  public double getRecall(int clazz) {
    int tp = getTruePositives(clazz);
    int fn = getFalseNegatives(clazz);
    if (tp + fn == 0) {
      return 0;
    }
    return (double) tp / (tp + fn);
  }

  public double getF1Measure(int clazz) {
    double p = getPrecision(clazz);
    double r = getRecall(clazz);
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
    double p = getMacroPrecision();
    double r = getMacroRecall();
    if (p + r == 0) {
      return 0;
    }
    return 2 * p * r / (p + r);
  }

  public double getMicroPrecision() {
    int tps = 0;
    int fps = 0;
    for (int i = 0; i < counts.length; i++) {
      tps += getTruePositives(i);
      fps += getFalsePositives(i);
    }
    if (tps + fps == 0) {
      return 0;
    }
    return (double) tps / (tps + fps);
  }

  public double getCohenKappa() {
    int[] sumRows = new int[counts.length];
    int[] sumColumns = new int[counts.length];
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

  public int getTruePositives(int clazz) {
    return counts[clazz][clazz];
  }

  public int getTrueNegatives(int clazz) {
    int tn = 0;
    for (int i = 0; i < counts.length; i++) {
      for (int j = 0; j < counts.length; j++) {
        if (i != clazz && j != clazz) {
          tn += counts[i][j];
        }
      }
    }
    return tn;
  }

  public int getFalsePositives(int clazz) {
    int fp = 0;
    for (int i = 0; i < counts.length; i++) {
      if (i != clazz) {
        fp += counts[i][clazz];
      }
    }
    return fp;
  }

  public int getFalseNegatives(int clazz) {
    int fn = 0;
    for (int i = 0; i < counts.length; i++) {
      if (i != clazz) {
        fn += counts[clazz][i];
      }
    }
    return fn;
  }

  public void add(final ConfusionMatrix confusionMatrix) {
    for (int i = 0; i < counts.length; i++) {
      for (int j = 0; j < counts[i].length; j++) {
        counts[i][j] += confusionMatrix.counts[i][j];
      }
    }
  }

  public int getNumClasses() {
    return counts.length;
  }

  public String toSummaryString() {
    String s = "=== Summary ===\n";
    s += "Cohen's kappa: " + getCohenKappa() + '\n';
    s += "Macro precision: " + getMacroPrecision() + '\n';
    s += "Macro recall: " + getMacroRecall() + '\n';
    s += "Macro f-measure: " + getMacroF1Measure() + '\n';
    s += "Micro precision: " + getMicroPrecision() + "\n";
    return s;
  }

  public String toClassDetailsString() {
    final StringBuilder sb = new StringBuilder("=== Detailed Accuracy By Class ===\n");
    sb.append("class\tprecision\trecall\tf1-measure\n");
    for (int i = 0; i < counts.length; i++) {
      sb.append(i).append("\t").append(getPrecision(i)).append("\t").append(getRecall(i)).append("\t").append(getF1Measure(i)).append("\n");
    }
    return sb.toString();
  }

  @Override
  public String toString() {
    final StringBuilder sb = new StringBuilder("=== Confusion Matrix ===\n");
    for (int i = 0; i < counts.length; i++) {
      for (int j = 0; j < counts.length; j++) {
        sb.append(counts[i][j]).append("\t");
      }
      sb.append("\n");
    }
    return sb.toString();
  }

}
