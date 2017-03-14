package com.spbsu.ml.methods.multiclass.gradfac;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.VecOptimization;

/**
 * User: qdeee
 * Date: 31.03.15
 */
public class MultiClassColumnBootstrapOptimization implements VecOptimization<L2> {
  private final FastRandom random;
  private final VecOptimization<L2> strong;
  private final double poisonMeanFreq;

  public MultiClassColumnBootstrapOptimization(final VecOptimization<L2> strong, final FastRandom random, final double poisonMeanFreq) {
    this.random = random;
    this.strong = strong;
    this.poisonMeanFreq = poisonMeanFreq;
  }

  @Override
  public Trans fit(final VecDataSet learn, final L2 mlllogitGrad) {
    final Mx grad = (Mx) mlllogitGrad.target;

    final Mx reweightedGrad = VecTools.copy(grad);
    for (int j = 0; j < reweightedGrad.columns(); j++) {
      VecTools.scale(reweightedGrad.col(j), random.nextPoisson(poisonMeanFreq));
    }
    final L2 reweightedLoss = DataTools.newTarget(L2.class, reweightedGrad, learn);
    return strong.fit(learn, reweightedLoss);
  }
}
