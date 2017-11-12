package com.expleague.ml.distributions.parametric;

import com.expleague.ml.distributions.DynamicRandomVec;
import com.expleague.ml.distributions.RandomVec;

/**
 * Created by noxoomo on 22/10/2017.
 */

public interface BetaVecDistribution extends DynamicRandomVec<BetaDistribution> {

  double alpha(final int idx);

  double beta(final int idx);

  BetaVecDistribution update(final int idx, final double alpha, final double beta);

  default BetaVecDistribution update(final int idx, final BetaDistribution distribution) {
    return update(idx, distribution.alpha(), distribution.beta());
  }
}
