package com.expleague.ml.distributions.bayesianUpdaters;
//
//import com.expleague.ml.bayesianEstimation.ConjugateBayesianEstimator;
//import com.expleague.ml.distributions.parametric.NormalGammaDistribution;
//import com.expleague.ml.distributions.parametric.NormalGammaVecDistribution;
//import com.expleague.ml.distributions.parametric.impl.NormalGammaDistributionImpl;
//
//import static com.expleague.commons.math.MathTools.sqr;
//
//public class NormalGammaNormalUpdater implements ConjugateBayesianEstimator<NormalGammaDistribution> {
//
//
//  @Override
//  public NormalGammaDistribution improperPrior() {
//    return new NormalGammaDistributionImpl(0, 0, 0, 0);
//  }
//
//  @Override
//  public int dim() {
//    return 4;
//  }
//
//  //TODO: awfully slow way, current implementation only for readability
//  @Override
//  public RandomVec<NormalGammaDistribution> update(final int idx,
//                                                   final double observation,
//                                                   final RandomVec<NormalGammaDistribution> distribution) {
//    if (distribution instanceof NormalGammaVecDistribution) {
//      NormalGammaVecDistribution dist = (NormalGammaVecDistribution) distribution;
//      final double mu = (dist.lambda(idx) * dist.mu(idx) + observation) / (dist.lambda(idx)  + 1);
//      final double lambda = dist.lambda(idx) + 1;
//      final double alpha = dist.alpha(idx) + 1.0 / 2;
//      final double beta = dist.beta(idx) +  dist.lambda(idx) * sqr(observation - dist.mu(idx)) / lambda / 2;
//
//      dist.update(idx, mu, lambda, alpha, beta);
//    } else {
//      final NormalGammaDistribution coordinate = distribution.randomVariable(idx);
//      update(observation, coordinate);
//    }
//    return distribution;
//  }
//
//  @Override
//  public NormalGammaDistribution update(final double observation, final NormalGammaDistribution coordinate) {
//    final double mu = (coordinate.lambda() * coordinate.mu() + observation) / (coordinate.lambda()  + 1);
//    final double lambda = coordinate.lambda() + 1;
//    final double alpha = coordinate.alpha() + 1.0 / 2;
//    final double beta = coordinate.beta() +  coordinate.lambda() * sqr(observation - coordinate.mu()) / lambda / 2;
//    return coordinate.update(mu, lambda, alpha, beta);
//  }
//
//}
