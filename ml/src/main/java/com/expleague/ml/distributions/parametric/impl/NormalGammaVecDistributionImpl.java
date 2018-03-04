//package com.expleague.ml.distributions.parametric.impl;
//
//import com.expleague.commons.func.Action;
//import com.expleague.commons.random.FastRandom;
//import com.expleague.ml.distributions.RandomVecBuilder;
//import com.expleague.ml.distributions.parametric.*;
//import com.expleague.ml.distributions.samplers.RandomVecSampler;
//import gnu.trove.list.array.TFloatArrayList;
//
///**
// * Created by noxoomo on 27/10/2017.
// */
//public class NormalGammaVecDistributionImpl extends RandomVec.IndependentCoordinatesDistribution<NormalGammaDistribution> implements NormalGammaVecDistribution {
//  private final SamplerImpl sampler = new SamplerImpl();
//
//  private final TFloatArrayList means = new TFloatArrayList();
//  private final TFloatArrayList lambdas = new TFloatArrayList();
//  private final TFloatArrayList alphas = new TFloatArrayList();
//  private final TFloatArrayList betas = new TFloatArrayList();
//
//
//  public NormalGammaVecDistributionImpl() {
//  }
//
//
//  @Override
//  public RandomVecSampler sampler() {
//    return sampler;
//  }
//
//  @Override
//  public double expectation(final int idx) {
//    return means.get(idx);
//  }
//
//  @Override
//  public double cumulativeProbability(final int idx, final double x) {
//    throw new RuntimeException("Unimplemeented");
//  }
//
//  @Override
//  public NormalGammaDistribution randomVariable(final int idx) {
//    return new CoordinateImpl(this, idx);
//  }
//
//  @Override
//  public RandomVec<NormalGammaDistribution> setRandomVariable(final int idx, final NormalGammaDistribution var) {
//    means.set(idx, (float) var.mu());
//    lambdas.set(idx, (float) var.lambda());
//    alphas.set(idx, (float) var.alpha());
//    betas.set(idx, (float) var.beta());
//    return this;
//  }
//
//
//  @Override
//  public int dim() {
//    return means.size();
//  }
//
//
//  @Override
//  public RandomVecBuilder<NormalGammaDistribution> builder() {
//    return new Builder();
//  }
//
//  @Override
//  public Action<NormalGammaDistribution> updater() {
//    return updater;
//  }
//
//  @Override
//  public double mu(final int idx) {
//    return means.get(idx);
//  }
//
//  @Override
//  public double lambda(final int idx) {
//    return lambdas.get(idx);
//  }
//
//  @Override
//  public double alpha(final int idx) {
//    return alphas.get(idx);
//  }
//
//  @Override
//  public double beta(final int idx) {
//    return betas.get(idx);
//  }
//
//  @Override
//  public NormalGammaVecDistribution update(final int idx,
//                                           final double mu, final double lambda, final double alpha, final double beta) {
//    means.set(idx, (float) mu);
//    lambdas.set(idx, (float) lambda);
//    alphas.set(idx, (float) alpha);
//    betas.set(idx, (float) beta);
//    return this;
//  }
//
//
//  protected class SamplerImpl implements RandomVecSampler {
//    @Override
//    public final double instance(final FastRandom random, final int i) {
//      final double alpha = alphas.get(i);
//      final double beta = betas.get(i);
//      final double mu = means.get(i);
//      final double lambda = lambdas.get(i);
//      return NormalGammaDistribution.Stub.instance(random, mu, lambda, alpha, beta);
//    }
//
//    @Override
//    public final int dim() {
//      return alphas.size();
//    }
//  }
//
//  private void add(final NormalGammaDistribution distribution) {
//    means.add((float) distribution.mu());
//    lambdas.add((float) distribution.lambda());
//    alphas.add((float) distribution.alpha());
//    betas.add((float) distribution.beta());
//  }
//
//  private Action<NormalGammaDistribution> updater = this::add;
//
//  public static class Builder implements RandomVecBuilder<NormalGammaDistribution> {
//    private final NormalGammaVecDistributionImpl impl;
//
//    public Builder() {
//      impl = new NormalGammaVecDistributionImpl();
//    }
//
//    public final Builder add(final NormalGammaDistribution distribution) {
//      impl.add(distribution);
//      return this;
//    }
//
//    @Override
//    public final RandomVec<NormalGammaDistribution> build() {
//      return impl;
//    }
//  }
//
//
//  protected class CoordinateImpl extends CoordinateProjectionStub<NormalGammaVecDistributionImpl> implements NormalGammaDistribution {
//
//    protected CoordinateImpl(final NormalGammaVecDistributionImpl featureBinarization, final int idx) {
//      super(featureBinarization, idx);
//    }
//
//    @Override
//    public double mu() {
//      return featureBinarization.means.get(idx);
//    }
//
//    @Override
//    public double lambda() {
//      return featureBinarization.lambdas.get(idx);
//    }
//
//    @Override
//    public final double alpha() {
//      return featureBinarization.alphas.get(idx);
//    }
//
//    @Override
//    public final double beta() {
//      return featureBinarization.betas.get(idx);
//    }
//
//    @Override
//    public final NormalGammaDistribution update(final double mu, final double lambda,
//                                                final double alpha, final double beta) {
//      featureBinarization.update(idx, mu, lambda, alpha, beta);
//      return this;
//    }
//
//    public final boolean equals(final Object o) {
//      if (this == o) return true;
//      if (!(o instanceof NormalGammaDistribution)) return false;
//      return Stub.equals(this, (NormalGammaDistribution) o);
//    }
//
//    @Override
//    public final int hashCode() {
//      return Stub.hashCode(this);
//    }
//
//    @Override
//    public RandomVecBuilder<NormalGammaDistribution> vecBuilder() {
//      return new Builder();
//    }
//  }
//}
