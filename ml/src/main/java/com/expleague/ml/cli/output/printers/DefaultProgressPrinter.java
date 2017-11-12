package com.expleague.ml.cli.output.printers;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.BinModelWithGrid;
import com.expleague.ml.Binarize;
import com.expleague.ml.ProgressHandler;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.randomnessAware.RandomnessAwareTrans;

/**
 * User: qdeee
 * Date: 04.09.14
 */
public class DefaultProgressPrinter implements ProgressHandler {
  private final Func loss;
  private final Func[] testMetrics;
  private final int printPeriod;
  private Vec learnValues;
  private final Vec[] testValuesArray;
  private VecDataSet learnDs;
  private VecDataSet testDs;

  public DefaultProgressPrinter(final Pool learn, final Pool test, final Func learnMetric, final Func[] testMetrics, final int printPeriod) {
    this.loss = learnMetric;
    this.testMetrics = testMetrics;
    this.printPeriod = printPeriod;
    learnValues = new ArrayVec(learnMetric.xdim());
    testValuesArray = new Vec[testMetrics.length];
    for (int i = 0; i < testValuesArray.length; i++) {
      testValuesArray[i] = new ArrayVec(testMetrics[i].xdim());
    }
    this.learnDs = learn.vecData();
    this.testDs = test.vecData();
  }

  int iteration = 0;

  @Override
  public void invoke(final Trans partial) {
    iteration++;

    if (partial instanceof Ensemble) {
      final Ensemble ensemble = (Ensemble) partial;
      final double step = ensemble.wlast();
      final Trans last = ensemble.last();

      final Mx learnTrans;
      final Mx testTrans;

//      if (last instanceof BinModelWithGrid) {
//        BinModelWithGrid model = (BinModelWithGrid) last;
//        BinarizedDataSet learnSet = learnDs.cache().cache(Binarize.class, VecDataSet.class).binarize(model.grid());
//        BinarizedDataSet testSet = testDs.cache().cache(Binarize.class, VecDataSet.class).binarize(model.grid());
//        learnTrans = model.transAll(learnSet);
//        testTrans = model.transAll(testSet);
//      }
//      else
      if (last instanceof RandomnessAwareTrans) {
        learnTrans = ((RandomnessAwareTrans) last).transAll(learnDs);
        testTrans = ((RandomnessAwareTrans) last).transAll(testDs);
      } else {
        learnTrans = last.transAll(learnDs.data());
        testTrans = last.transAll(testDs.data());
      }

      VecTools.append(learnValues, VecTools.scale(learnTrans, step));
      final Mx testEvaluation = VecTools.scale(testTrans, step);
      for (int t = 0; t < testValuesArray.length; ++t) {
        VecTools.append(testValuesArray[t], testEvaluation);
      }

    }
    else if (iteration % printPeriod == 0)

    {
      learnValues = partial.transAll(learnDs.data());
      final Mx testEvaluate = partial.transAll(testDs.data());
      for (int i = 0; i < testValuesArray.length; i++) {
        testValuesArray[i] = testEvaluate;
      }
    }

    if (iteration % printPeriod != 0)

    {
      return;
    }

    System.out.print(iteration);
    System.out.print("\t" + loss.value(learnValues));
    for (
        int i = 0;
        i < testMetrics.length; i++)

    {
      System.out.print("\t" + testMetrics[i].value(testValuesArray[i]));
    }
    System.out.println();
  }
}
