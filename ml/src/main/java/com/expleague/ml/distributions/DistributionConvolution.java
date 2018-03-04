package com.expleague.ml.distributions;

public interface DistributionConvolution {

  RandomVec empty(int dim);
  RandomVariable empty();

  RandomVec combine(final RandomVariable value, final double valueScale,
                    final int idx, final RandomVec to, final double toScale);

  RandomVec combine(final RandomVec value, final double valueScale,
                    final RandomVec to, final double toScale);

  RandomVariable combine(final RandomVariable value, final double valueScale,
                         final RandomVariable to, final double toScale);
}
