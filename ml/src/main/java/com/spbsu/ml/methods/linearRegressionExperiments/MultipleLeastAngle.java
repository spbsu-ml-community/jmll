package com.spbsu.ml.methods.linearRegressionExperiments;

import com.spbsu.commons.util.ThreadTools;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.MultipleVecOptimization;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * Created by noxoomo on 10/06/15.
 */
public class MultipleLeastAngle extends MultipleVecOptimization.Stub<L2> {

  private static ThreadPoolExecutor exec = ThreadTools.createBGExecutor("ridge regressions", -1);
  final WeakLeastAngle regression = new WeakLeastAngle();

  @Override
  public Trans[] fit(VecDataSet[] learn, L2[] loss) {
    if (learn.length != loss.length)
      throw new IllegalArgumentException("losses count ≠ ds count");

    final boolean[] empty = new boolean[learn.length];

    final List<VecDataSet> datas = new ArrayList<>(loss.length);
    final List<L2> targets = new ArrayList<>(loss.length);

    int featureCount = 0;
    WeakLeastAngle.WeakLinear zeroWeight = null;
    for (int i = 0; i < learn.length; ++i) {
      if (loss[i] == null || loss[i].dim() + 1 < 3) {
        empty[i] = true;
      } else {
        final VecDataSet data = learn[i];
        final L2 target = loss[i];
        featureCount = data.xdim();
        datas.add(data);
        targets.add(target);
      }
    }

    final WeakLeastAngle.WeakLinear[] result = new WeakLeastAngle.WeakLinear[datas.size()];

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

    WeakLeastAngle.WeakLinear[] totalResult = new  WeakLeastAngle.WeakLinear[empty.length];
    int ind = 0;
    for (int i = 0; i < empty.length; ++i) {
      if (empty[i]) {
        if (zeroWeight == null) {
          zeroWeight = new WeakLeastAngle.WeakLinear(featureCount,0,0);
        }
        totalResult[i] = zeroWeight;
      } else {
        totalResult[i] = result[ind++];
      }
    }
    return totalResult;
  }
}


