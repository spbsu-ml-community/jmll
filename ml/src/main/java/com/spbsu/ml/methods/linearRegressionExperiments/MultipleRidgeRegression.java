package com.spbsu.ml.methods.linearRegressionExperiments;

import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.func.Linear;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.MultipleVecOptimization;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * Created by noxoomo on 10/06/15.
 */
public class MultipleRidgeRegression extends MultipleVecOptimization.Stub<L2> {
  private final double lambda;

  public MultipleRidgeRegression(double lambda) {
    this.lambda = lambda;
  }

  private static ThreadPoolExecutor exec = ThreadTools.createBGExecutor("ridge regressions", -1);

  @Override
  public Linear[] fit(VecDataSet[] learn, L2[] loss) {
    if (learn.length != loss.length)
      throw new IllegalArgumentException("losses count ≠ ds count");

    final boolean[] empty = new boolean[learn.length];

    final List<VecDataSet> datas = new ArrayList<>(loss.length);
    final List<L2> targets = new ArrayList<>(loss.length);

    int featureCount = 0;
    Linear zeroWeight = null;
    for (int i = 0; i < learn.length; ++i) {
      if (loss[i] == null || loss[i].dim()  <  learn[i].xdim()) {
        empty[i] = true;
      } else {
        final VecDataSet data = learn[i];
        final L2 target = loss[i];
        featureCount = data.xdim();
        datas.add(data);
        targets.add(target);
      }
    }

    final RidgeRegression regression = new RidgeRegression(lambda);
    final Linear[] result = new Linear[datas.size()];

    final CountDownLatch latch = new CountDownLatch(datas.size());

    for (int i = 0; i < result.length; ++i) {
      final int ind = i;
      exec.submit(new Runnable() {
        @Override
        public void run() {
          result[ind] = regression.fit(datas.get(ind), targets.get(ind));
          latch.countDown();
        }
      });
    }

    try {
      latch.await();
    } catch (InterruptedException e) {
      e.printStackTrace();
    }

    Linear[] totalResult = new Linear[empty.length];
    int ind = 0;
    for (int i = 0; i < empty.length; ++i) {
      if (empty[i]) {
        if (zeroWeight == null) {
          zeroWeight = new Linear(new double[featureCount]);
        }
        totalResult[i] = zeroWeight;
      } else {
        totalResult[i] = result[ind++];
      }
    }
    return totalResult;
  }
}


