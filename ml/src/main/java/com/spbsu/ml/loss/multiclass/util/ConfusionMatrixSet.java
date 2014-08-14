package com.spbsu.ml.loss.multiclass.util;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;


import com.spbsu.commons.func.Evaluator;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;

/**
 * User: amosov-f
 * Date: 14.08.14
 * Time: 12:29
 */
public class ConfusionMatrixSet {

  private final List<ConfusionMatrix> confusionMatrices;
  private int numClasses = -1;

  private final StandardDeviation standardDeviation = new StandardDeviation();

  public ConfusionMatrixSet() {
    confusionMatrices = new ArrayList<>();
  }

  public ConfusionMatrixSet(final List<ConfusionMatrix> confusionMatrices) {
    final Set<Integer> numClassesSet = new HashSet<>();
    for (final ConfusionMatrix matrix : confusionMatrices) {
      numClassesSet.add(matrix.getNumClasses());
    }
    if (numClassesSet.size() > 1) {
      throw new IllegalArgumentException("Sizes of matrices must be equal");
    }
    if (!confusionMatrices.isEmpty()) {
      numClasses = confusionMatrices.get(0).getNumClasses();
    }
    this.confusionMatrices = new ArrayList<>(confusionMatrices);
  }

  public void add(final ConfusionMatrix matrix) {
    if (numClasses == -1) {
      numClasses = matrix.getNumClasses();
    }
    confusionMatrices.add(matrix);
  }

  public double getStdDeviation(final Evaluator<ConfusionMatrix> measureEvaluator) {
    final double[] microF1Measures = new double[confusionMatrices.size()];
    for (int i = 0; i < confusionMatrices.size(); i++) {
      microF1Measures[i] = measureEvaluator.value(confusionMatrices.get(i));
    }
    return standardDeviation.evaluate(microF1Measures);
  }

  public ConfusionMatrix merge() {
    final ConfusionMatrix merge = new ConfusionMatrix(numClasses);
    for (final ConfusionMatrix matrix : confusionMatrices) {
      merge.add(matrix);
    }
    return merge;
  }

  public String toSummaryString() {
    String s = "=== Confusion Matrices ===\n";
    s += "Cohen kappa std deviation: " + getStdDeviation(new CohenKappaEvaluator()) + "\n";
    s += "Macro precision std deviation: " + getStdDeviation(new MacroPrecisionEvaluator()) + "\n";
    s += "Macro recall std deviation: " + getStdDeviation(new MacroRecallEvaluator()) + "\n";
    s += "Macro f1-measure std deviation: " + getStdDeviation(new MacroF1MeasureEvaluator()) + "\n";
    s += "Micro precision std deviation: " + getStdDeviation(new MicroPrecisionEvaluator()) + "\n";
    return s;
  }

  public static class CohenKappaEvaluator implements Evaluator<ConfusionMatrix> {
    @Override
    public double value(final ConfusionMatrix matrix) {
      return matrix.getCohenKappa();
    }
  }

  public static class MacroPrecisionEvaluator implements Evaluator<ConfusionMatrix> {
    @Override
    public double value(final ConfusionMatrix matrix) {
      return matrix.getMacroPrecision();
    }
  }

  public static class MacroRecallEvaluator implements Evaluator<ConfusionMatrix> {
    @Override
    public double value(final ConfusionMatrix matrix) {
      return matrix.getMacroRecall();
    }
  }

  public static class MacroF1MeasureEvaluator implements Evaluator<ConfusionMatrix> {
    @Override
    public double value(final ConfusionMatrix matrix) {
      return matrix.getMacroF1Measure();
    }
  }

  public static class MicroPrecisionEvaluator implements Evaluator<ConfusionMatrix> {
    @Override
    public double value(final ConfusionMatrix matrix) {
      return matrix.getMicroPrecision();
    }
  }

}
