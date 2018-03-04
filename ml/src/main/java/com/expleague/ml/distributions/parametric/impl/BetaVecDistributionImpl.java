package com.expleague.ml.distributions.parametric.impl;

import com.expleague.commons.random.FastRandom;
import com.expleague.ml.distributions.*;
import com.expleague.ml.distributions.parametric.BetaDistribution;
import com.expleague.ml.distributions.parametric.BetaVecDistribution;
import gnu.trove.list.array.TFloatArrayList;

/**
 * Created by noxoomo on 27/10/2017.
 * partially based on apacheMath. we need vectorized version
 */
public class BetaVecDistributionImpl extends RandomVec.CoordinateIndependentStub implements BetaVecDistribution {
  private final TFloatArrayList alphas;
  private final TFloatArrayList betas;
  private final TFloatArrayList z;

  public BetaVecDistributionImpl() {
    alphas = new TFloatArrayList();
    betas = new TFloatArrayList();
    z = new TFloatArrayList();
  }


  private double lazyGetZ(int i) {
    if (z.size() <= i) {
      for (int k = z.size(); k <= i; ++k) {
        z.add(Float.NaN);
      }
    }

    if (Float.isNaN(z.get(i))) {
      final double alpha = alphas.get(i);
      final double beta = betas.get(i);
      z.set(i, (float) BetaDistribution.Stub.computeZ(alpha, beta));
    }
    return z.get(i);
  }

  public double expectation(final int idx) {
    final double alpha = alphas.get(idx);
    final double beta = betas.get(idx);
    return BetaDistribution.Stub.expectation(alpha, beta);
  }


  @Override
  public double cdf(final int idx, double x) {
    final double alpha = alphas.get(idx);
    final double beta = betas.get(idx);
    return BetaDistribution.Stub.cumulativeProbability(x, alpha, beta);
  }


  public double logDensity(final int idx, final double x) {
    this.lazyGetZ(idx);
    final double z = lazyGetZ(idx);
    final double alpha = alphas.get(idx);
    final double beta = betas.get(idx);

    return BetaDistribution.Stub.logDensity(x, alpha, beta, z);
  }

  public RandomVecBuilder<BetaDistribution> builder() {
    return new Builder();
  }


  @Override
  public double alpha(final int idx) {
    return alphas.get(idx);
  }

  @Override
  public double beta(final int idx) {
    return betas.get(idx);
  }

  public BetaVecDistribution update(final int idx, final double alpha, final double beta) {
    alphas.set(idx, (float) alpha);
    betas.set(idx, (float) beta);
    return this;
  }

  @Override
  public BetaDistribution at(int idx) {
    return new CoordinateImpl(this, idx);
  }

  @Override
  public int length() {
    return alphas.size();
  }

  @Override
  public double instance(int idx, FastRandom random) {
    return BetaDistribution.Stub.instance(random, alphas.get(idx), betas.get(idx));
  }


  private void add(final BetaDistribution distribution) {
    if (distribution.alpha() <0 || distribution.beta() < 0) {
      throw new RuntimeException("err");
    }
    alphas.add((float) distribution.alpha());
    betas.add((float) distribution.beta());
  }

  public static class Builder implements RandomVecBuilder<BetaDistribution> {
    private final BetaVecDistributionImpl impl;

    public Builder() {
      impl = new BetaVecDistributionImpl();
    }

    public final Builder add(final BetaDistribution distribution) {
      impl.add(distribution);
      return this;
    }


    @Override
    public final BetaVecDistribution build() {
      return impl;
    }
  }

  public static class BetaDistributionList implements RandomList<BetaDistribution> {
    private Builder builder = new Builder();

    @Override
    public BetaDistribution get(int idx) {
      return builder.impl.at(idx);
    }

    @Override
    public void set(int idx, final BetaDistribution distribution) {
      builder.impl.alphas.set(idx, (float) distribution.alpha());
      builder.impl.betas.set(idx, (float) distribution.beta());
      builder.impl.z.set(idx, Float.NaN);
    }

    @Override
    public RandomVecBuilder<BetaDistribution> add(BetaDistribution dist) {
      builder.add(dist);
      return null;
    }

    @Override
    public RandomVec build() {
      return builder.build();
    }
  }

  protected class CoordinateImpl extends RandomVec.CoordinateProjectionStub<BetaVecDistributionImpl> implements BetaDistribution {

    protected CoordinateImpl(final BetaVecDistributionImpl owner,
                             final int idx) {
      super(owner, idx);
    }

    @Override
    public final double alpha() {
      return alphas.get(idx);
    }

    @Override
    public final double beta() {
      return betas.get(idx);
    }

    public final BetaDistribution update(final double alpha,
                                         final double beta) {
      alphas.set(idx, (float) alpha);
      betas.set(idx, (float) beta);
      return this;
    }

    public final  boolean equals(final Object o) {
      if (this == o) return true;
      if (!(o instanceof BetaDistribution)) return false;
      return BetaDistribution.Stub.equals(this, (BetaDistribution) o);
    }

    @Override
    public final int hashCode() {
      return BetaDistribution.Stub.hashCode(this);
    }
  }
}
