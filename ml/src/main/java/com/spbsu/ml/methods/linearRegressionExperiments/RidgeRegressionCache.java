package com.spbsu.ml.methods.linearRegressionExperiments;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.func.Linear;
import com.spbsu.ml.loss.L2;

/**
 * Created by noxoomo on 10/06/15.
 */
public class RidgeRegressionCache {
  private final Mx cov;
  private final Vec covTargetWithFeatures;

  public RidgeRegressionCache(VecDataSet learn, L2 l2) {
    Vec target = l2.target;
    Mx data = learn.data();
    cov = new VecBasedMx(data.columns(), data.columns());
    covTargetWithFeatures = new ArrayVec(data.columns());

    for (int i = 0; i < data.columns(); ++i) {
      final Vec feature = data.col(i);
      cov.set(i, i, VecTools.multiply(feature, feature));
      covTargetWithFeatures.set(i, VecTools.multiply(feature, target));
      for (int j = i + 1; j < data.columns(); ++j) {
        final double val = VecTools.multiply(feature, data.col(j));
        cov.set(i, j, val);
        cov.set(j, i, val);
      }
    }
  }

  public RidgeRegressionCache(Mx cov, Vec covTargetWithFeatures) {
    this.cov = cov;
    this.covTargetWithFeatures = covTargetWithFeatures;
  }

  public Linear fit(final double alpha) {
    final Mx regCov = new VecBasedMx(cov);
    for (int i = 0; i < regCov.columns(); ++i)
      regCov.adjust(i, i, alpha);
    final Mx invCov = MxTools.inverse(regCov);
    final Vec weights = MxTools.multiply(invCov, covTargetWithFeatures);
    return new Linear(weights);
  }

}
