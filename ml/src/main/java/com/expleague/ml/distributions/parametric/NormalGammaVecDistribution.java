package com.expleague.ml.distributions.parametric;

import com.expleague.ml.distributions.VecDistribution;

/**
 * Created by noxoomo on 22/10/2017.
 */

public interface NormalGammaVecDistribution extends VecDistribution {

  double mu(final int idx);
  double lambda(final int idx);
  double alpha(final int idx);
  double beta(final int idx);

  NormalGammaVecDistribution update(final int idx,
                                    final double mu, final double lambda,
                                    final double alpha, final double beta);

  default NormalGammaVecDistribution update(final int idx,
                                            final NormalGammaDistribution distribution) {
    return update(idx, distribution.mu(), distribution.lambda(), distribution.alpha(), distribution.beta());
  }
}
