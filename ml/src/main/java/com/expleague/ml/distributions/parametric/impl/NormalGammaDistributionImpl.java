//package com.expleague.ml.distributions.parametric.impl;
//
//import com.expleague.commons.random.FastRandom;
//import com.expleague.ml.distributions.RandomVecBuilder;
//import com.expleague.ml.distributions.parametric.NormalGammaDistribution;
//
///**
// * Created by noxoomo on 27/10/2017.
// */
//public class NormalGammaDistributionImpl implements NormalGammaDistribution {
//
//  private double mu;
//  private double lambda;
//  private double alpha;
//  private double beta;
//
//  public NormalGammaDistributionImpl(final double mu, final double lambda, final double alpha, final double beta) {
//    this.mu = mu;
//    this.lambda = lambda;
//    this.alpha = alpha;
//    this.beta = beta;
//  }
//
//
//  @Override
//  public double mu() {
//    return mu;
//  }
//
//  @Override
//  public double lambda() {
//    return lambda;
//  }
//
//  @Override
//  public double alpha() {
//    return alpha;
//  }
//
//  @Override
//  public double beta() {
//    return beta;
//  }
//
//  @Override
//  public NormalGammaDistribution update(final double mu, final double lambda, final double alpha, final double beta) {
//    this.mu = mu;
//    this.lambda = lambda;
//    this.alpha = alpha;
//    this.beta = beta;
//    return this;
//  }
//
//  @Override
//  public double cdf(final double value) {
//    throw new RuntimeException("Unimplemented");
//  }
//
//  @Override
//  public double logDensity(double value) {
//    throw new RuntimeException("Unimplemented");
//  }
//
//  @Override
//  public double sample(FastRandom random) {
//    return NormalGammaDistribution.Stub.instance(random, mu, lambda, alpha, beta);
//  }
//}
