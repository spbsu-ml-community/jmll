package com.expleague.ml.cli.output.printers;

import com.expleague.commons.func.Action;
import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.ProgressHandler;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.distributions.parametric.NormalDistribution;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.func.RandomFuncEnsemble;
import com.expleague.ml.loss.L2;
import com.expleague.ml.methods.RandomnessAwareGradientBoosting;
import com.expleague.ml.randomnessAware.RandomFunc;

/**
 * User: noxoomo
 * Date: 04.09.14
 */
public class RandomnessAwareProgressPrinter implements Action<RandomFunc> {
  private final Func loss;
  private final Func[] testMetrics;
  private final int printPeriod;

  private RandomVec learnCursor;
  private RandomVec testCursor;

  private VecDataSet learnDs;
  private VecDataSet testDs;

  public RandomnessAwareProgressPrinter(final Pool learn, final Pool test,
                                        final Func learnMetric, final Func[] testMetrics,
                                        final int printPeriod) {
    this.loss = learnMetric;
    this.testMetrics = testMetrics;
    this.printPeriod = printPeriod;

    this.learnDs = learn.vecData();
    this.testDs = test.vecData();
  }

  int iteration = 0;

  @Override
  public void invoke(final RandomFunc partial) {
    iteration++;

    if (partial instanceof RandomFuncEnsemble) {
      final RandomFuncEnsemble ensemble = (RandomFuncEnsemble) partial;
      final double step = ensemble.wlast();
      final RandomFunc last = ensemble.last();

      if (learnCursor == null) {
        learnCursor = last.emptyVec(learnDs.data().rows());
      }
      if (testCursor == null && testDs != null) {
        testCursor = last.emptyVec(testDs.data().rows());
      }

      last.appendTo(step, learnDs, learnCursor);
      if (testCursor != null) {
        last.appendTo(step, testDs, testCursor);
      }
    }
    else {
      throw new RuntimeException("Wrong partial type");
    }

    if (iteration % printPeriod != 0) {
      return;
    }

    System.out.print(iteration);
    final Vec learnApprox = learnCursor.expectation();
    final Vec testApprox = testCursor.expectation();

    System.out.print("\t" + loss.value(learnApprox));
    for (int i = 0; i < testMetrics.length; i++) {
      System.out.print("\t" + testMetrics[i].value(testApprox));
    }
    System.out.println();
  }
}
