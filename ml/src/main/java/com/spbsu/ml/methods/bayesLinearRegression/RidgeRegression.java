package com.spbsu.ml.methods.bayesLinearRegression;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.func.BiasedLinear;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.VecOptimization;

/**
 * Created by noxoomo on 10/06/15.
 */
public class RidgeRegression implements VecOptimization<L2> {
  final double alpha;

  public RidgeRegression(double alpha) {
    this.alpha = alpha;
  }

  @Override
  public BiasedLinear fit(VecDataSet learn, L2 l2) {
    Vec target = l2.target;
    Mx data = learn.data();
    Mx cov = new VecBasedMx(data.columns() + 1, data.columns() + 1);
    Vec covTargetWithFeatures = new ArrayVec(data.columns() + 1);
    for (int i = 0; i < data.columns(); ++i) {
      final Vec feature = data.col(i);
      cov.set(i, i, VecTools.multiply(feature, feature));
      covTargetWithFeatures.set(i, VecTools.multiply(feature, target));
      for (int j = i + 1; j < data.columns(); ++j) {
        final double val = VecTools.multiply(feature, data.col(j));
        cov.set(i, j, val);
        cov.set(j, i, val);
      }
      final double sum = VecTools.sum(feature);
      cov.set(i, data.columns(), sum);
      cov.set(data.columns(), i, sum);
    }
    cov.set(data.columns(), data.columns(), data.rows());
    covTargetWithFeatures.set(data.columns(), VecTools.sum(target));
    for (int i = 0; i < cov.columns(); ++i) {
      cov.adjust(i, i, alpha);
    }
    Mx invCov = MxTools.inverse(cov);
    Vec weights = MxTools.multiply(invCov, covTargetWithFeatures);
    final double bias = weights.get(data.columns());
    final double[] w = new double[data.columns()];
    for (int i=0; i < w.length;++i) {
      w[i] = weights.get(i);
    }
    return new BiasedLinear(w, bias);
  }
}
