package com.spbsu.ml.cli.output.printers;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.Func;
import com.spbsu.ml.ProgressHandler;
import com.spbsu.ml.Trans;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.func.Ensemble;

import static com.spbsu.commons.math.vectors.VecTools.append;

/**
 * User: qdeee
 * Date: 04.09.14
 */
public class DefaultProgressPrinter implements ProgressHandler {
  private final Pool learn;
  private final Pool test;
  private final Func loss;
  private final Func[] testMetrics;
  private final int printPeriod;
  private Vec learnValues;
  private final Vec[] testValuesArray;

  public DefaultProgressPrinter(final Pool learn, final Pool test, final Func learnMetric, final Func[] testMetrics, final int printPeriod) {
    this.learn = learn;
    this.test = test;
    this.loss = learnMetric;
    this.testMetrics = testMetrics;
    this.printPeriod = printPeriod;
    learnValues = new ArrayVec(learnMetric.xdim());
    testValuesArray = new Vec[testMetrics.length];
    for (int i = 0; i < testValuesArray.length; i++) {
      testValuesArray[i] = new ArrayVec(testMetrics[i].xdim());
    }
  }

  int iteration = 0;

  @Override
  public void invoke(final Trans partial) {
    iteration++;

    if (partial instanceof Ensemble) {
      final Ensemble ensemble = (Ensemble) partial;
      final double step = ensemble.wlast();
      final Trans last = ensemble.last();

      append(learnValues, VecTools.scale(last.transAll(learn.vecData().data()), step));
      final Mx testEvaluation = VecTools.scale(last.transAll(test.vecData().data()), step);
      for (int t = 0; t < testValuesArray.length; ++t) {
        append(testValuesArray[t], testEvaluation);
      }

    } else if (iteration % printPeriod == 0) {
      learnValues = partial.transAll(learn.vecData().data());
      final Mx testEvaluate = partial.transAll(test.vecData().data());
      for (int i = 0; i < testValuesArray.length; i++) {
        testValuesArray[i] = testEvaluate;
      }
    }

    if (iteration % 10 != 0) {
      return;
    }

    System.out.print(iteration);
    System.out.print(" " + loss.value(learnValues));
    for (int i = 0; i < testMetrics.length; i++) {
      System.out.print("\t" + testMetrics[i].value(testValuesArray[i]));
    }
    System.out.println();
  }
}
