package com.spbsu.ml.cli.output.printers;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.Func;
import com.spbsu.ml.ProgressHandler;
import com.spbsu.ml.Trans;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.func.Ensemble;

import static com.spbsu.commons.math.vectors.VecTools.append;
import static com.spbsu.commons.math.vectors.VecTools.assign;

/**
 * User: qdeee
 * Date: 04.09.14
 */
public class DefaultProgressPrinter implements ProgressHandler {
  private final Pool learn;
  private final Pool test;
  private final Func loss;
  private final Func[] testMetrics;
  private Vec learnValues;
  private Vec[] testValuesArray;

  public DefaultProgressPrinter(Pool learn, Pool test, Func learnMetric, Func[] testMetrics) {
    this.learn = learn;
    this.test = test;
    this.loss = learnMetric;
    this.testMetrics = testMetrics;
    learnValues = new ArrayVec(learnMetric.xdim());
    testValuesArray = new ArrayVec[testMetrics.length];
    for (int i = 0; i < testValuesArray.length; i++) {
      testValuesArray[i] = new ArrayVec(testMetrics[i].xdim());
    }
  }

  int iteration = 0;

  @Override
  public void invoke(Trans partial) {
    if (partial instanceof Ensemble) {
      final Ensemble ensemble = (Ensemble) partial;
      final double step = ensemble.wlast();
      final Trans last = ensemble.last();

      append(learnValues, VecTools.scale(last.transAll(learn.vecData().data()), step));

      for (int t = 0; t < testValuesArray.length; ++t) {
        append(testValuesArray[t], VecTools.scale(last.transAll(test.vecData().data()), step));
      }
    }
    else {
      learnValues = partial.transAll(learn.vecData().data());
      for (int i = 0; i < testValuesArray.length; i++) {
        assign(testValuesArray[i], partial.transAll(test.vecData().data()));
      }
    }
    iteration++;
    if (iteration % 10 != 0) {
      return;
    }

    System.out.print(iteration);
    System.out.print(" " + loss.trans(learnValues));
    for (int i = 0; i < testMetrics.length; i++) {
      System.out.print("\t" + testMetrics[i].trans(testValuesArray[i]));
    }
    System.out.println();
  }
}
