package com.spbsu.ml.methods.linearRegressionExperiments;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.func.Linear;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.MultipleVecOptimization;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by noxoomo on 10/06/15.
 */
public class MultipleEbsRidgeRegression extends MultipleVecOptimization.Stub<L2> {
  @Override
  public Linear[] fit(VecDataSet[] learn, L2[] loss) {
    if (learn.length != loss.length)
      throw new IllegalArgumentException("losses count ≠ ds count");

    final boolean[] empty = new boolean[learn.length];

    final List<Mx> datas = new ArrayList<>(loss.length);
    final List<Vec> targets = new ArrayList<>(loss.length);

    int featureCount = 0;
    Linear zeroWeight = null;
    for (int i = 0; i < learn.length; ++i) {
      if (loss[i] == null || loss[i].dim() < learn[i].xdim()) {
        empty[i] = true;
      } else {
        final Mx data = learn[i].data();
        final Vec target = loss[i].target();
        featureCount = data.columns();
        datas.add(data);
        targets.add(target);
      }
    }

    final EmpericalBayesRidgeRegression regression = new EmpericalBayesRidgeRegression(
      datas.toArray(new Mx[datas.size()]),
      targets.toArray(new Vec[datas.size()]));
    Linear[] result = regression.fit();

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

