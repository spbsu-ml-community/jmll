package com.expleague.ml.distributions.parametric.impl;

import com.expleague.commons.func.Action;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.distributions.DynamicRandomVec;
import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.distributions.RandomVecBuilder;
import com.expleague.ml.distributions.parametric.BetaDistribution;
import com.expleague.ml.distributions.parametric.BetaVecDistribution;
import com.expleague.ml.distributions.samplers.RandomVecSampler;
import gnu.trove.list.array.TFloatArrayList;

import java.util.Arrays;

/**
 * Created by noxoomo on 27/10/2017.
 * partially based on apacheMath. we need vectorized version
 */
public class BetaVecDistributionImpl extends RandomVec.IndependentCoordinatesDistribution<BetaDistribution> implements BetaVecDistribution {
  private final SamplerImpl sampler = new SamplerImpl();
  private final TFloatArrayList alphas = new TFloatArrayList();
  private final TFloatArrayList betas = new TFloatArrayList();
  private TFloatArrayList z;

  public BetaVecDistributionImpl() {
    z = new TFloatArrayList();
  }

  private double lazyGetZ(int i) {
    for (int k = z.size(); k <= i; ++k) {
      z.set(k, Float.NaN);
    }
    if (Float.isNaN(z.get(i))) {
      final double alpha = alphas.get(i);
      final double beta = betas.get(i);
      z.set(i, (float) BetaDistribution.Stub.computeZ(alpha, beta));
    }
    return z.get(i);
  }

  @Override
  public RandomVecSampler sampler() {
    return sampler;
  }

  @Override
  public double expectation(final int idx) {
    final double alpha = alphas.get(idx);
    final double beta = betas.get(idx);
    return BetaDistribution.Stub.expectation(alpha, beta);
  }

  @Override
  public BetaDistribution randomVariable(final int idx) {
    return new CoordinateImpl(this, idx);
  }

  @Override
  public RandomVec<BetaDistribution> setRandomVariable(final int idx, final BetaDistribution var) {
    alphas.set(idx, (float) var.alpha());
    betas.set(idx, (float) var.beta());
    if (z.size() > idx) {
      z.set(idx, Float.NaN);
    }
    return this;
  }

  public double logDensity(final int idx, final Vec point) {
    return logDensity(idx, point.get(idx));
  }


  @Override
  public int dim() {
    return alphas.size();
  }

  @Override
  public double cumulativeProbability(final int idx, double x) {
    final double alpha = alphas.get(idx);
    final double beta = betas.get(idx);
    return BetaDistribution.Stub.cumulativeProbability(x, alpha, beta);
  }


  protected double logDensity(final int idx, final double x) {
    this.lazyGetZ(idx);
    final double z = lazyGetZ(idx);
    final double alpha = alphas.get(idx);
    final double beta = betas.get(idx);

    return BetaDistribution.Stub.logDensity(x, alpha, beta, z);
  }

  @Override
  public RandomVecBuilder<BetaDistribution> builder() {
    return new Builder();
  }

  @Override
  public Action<BetaDistribution> updater() {
    return updater;
  }

  @Override
  public double alpha(final int idx) {
    return alphas.get(idx);
  }

  @Override
  public double beta(final int idx) {
    return betas.get(idx);
  }

  @Override
  public BetaVecDistribution update(final int idx, final double alpha, final double beta) {
    alphas.set(idx, (float) alpha);
    betas.set(idx, (float) beta);
    return this;
  }


  protected class SamplerImpl implements RandomVecSampler {
    @Override
    public final  double instance(final FastRandom random, final int i) {
      final double alpha = alphas.get(i);
      final double beta = betas.get(i);
      return BetaDistribution.Stub.instance(random, alpha, beta);
    }

    @Override
    public final int dim() {
      return alphas.size();
    }
  }

  private void add(final BetaDistribution distribution) {
    alphas.add((float) distribution.alpha());
    betas.add((float) distribution.beta());
  }

  private Action<BetaDistribution> updater = this::add;

  public static class Builder implements RandomVecBuilder<BetaDistribution> {
    private final BetaVecDistributionImpl impl;

    public Builder() {
      impl = new BetaVecDistributionImpl();
    }

    public final  Builder add(final BetaDistribution distribution) {
      impl.add(distribution);
      return this;
    }

    @Override
    public final  RandomVec<BetaDistribution> build() {
      return impl;
    }
  }


  protected class CoordinateImpl extends IndependentCoordinatesDistribution<BetaDistribution>.CoordinateProjectionStub<BetaVecDistributionImpl> implements BetaDistribution {

    protected CoordinateImpl(final BetaVecDistributionImpl owner, final int idx) {
      super(owner, idx);
    }

    @Override
    public final double alpha() {
      return owner.alphas.get(idx);
    }

    @Override
    public final double beta() {
      return owner.betas.get(idx);
    }

    @Override
    public final BetaDistribution update(final double alpha, final double beta) {
      owner.alphas.set(idx, (float) alpha);
      owner.betas.set(idx, (float) beta);
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
