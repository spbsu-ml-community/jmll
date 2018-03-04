package com.expleague.ml.distributions;

import com.expleague.ml.distributions.parametric.NormalDistribution;
import com.expleague.ml.distributions.parametric.NormalDistributionImpl;
import com.expleague.ml.distributions.parametric.NormalVecDistribution;
import com.expleague.ml.distributions.parametric.impl.NormalVecDistributionImpl;

/**
 * Created by noxoomo on 12/02/2018.
 * TODO: combiner based on family type
 */
public class StupidOnlyNormalDistributionConvolution implements DistributionConvolution {
  @Override
  public RandomVec empty(int dim) {
    final NormalVecDistributionImpl.Builder builder = new NormalVecDistributionImpl.Builder();
    final NormalDistributionImpl degenerateNormal = new NormalDistributionImpl(0.0, 0.0);
    for (int i = 0; i < dim; ++i) {
      builder.add(degenerateNormal);
    }
    return builder.build();
  }

  @Override
  public RandomVariable empty() {
    return new NormalDistributionImpl(0, 0);
  }

  @Override
  public RandomVec combine(final RandomVariable value, double valueScale, int idx, RandomVec to, double toScale) {
    if (to instanceof NormalVecDistribution && value instanceof NormalDistribution) {
      ((NormalVecDistribution) to).scale(toScale).add(idx, valueScale, (NormalDistribution) value);
    }  else {
      throw new RuntimeException("Unimplemented yet " + to.getClass().getName() + " " + value.getClass().getName());
    }
    return to;
  }

  @Override
  public RandomVec combine(final RandomVec value, final double valueScale,
                           final RandomVec to, final double toScale) {
    if (to instanceof NormalVecDistribution && value instanceof NormalVecDistribution) {
      return ((NormalVecDistribution) to).scale(toScale).add((NormalVecDistribution) value, valueScale);
    }  else {
      throw new RuntimeException("Unimplemented yet");
    }
  }

  @Override
  public RandomVariable combine(RandomVariable value, double valueScale, RandomVariable to, double toScale) {
    if (to instanceof NormalDistribution && value instanceof NormalDistribution) {
      return ((NormalDistribution) to).scale(toScale).add((NormalDistribution) value, valueScale);
    }  else {
      throw new RuntimeException("Unimplemented yet");
    }
  }
}
