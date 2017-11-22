//package com.spbsu.ml.methods.greedyRegion;
//
//import com.spbsu.commons.sufficientSpaceExtractors.AdditiveStatistics;
//import com.spbsu.commons.math.vectors.Mx;
//import com.spbsu.commons.math.vectors.MxTools;
//import com.spbsu.commons.math.vectors.Vec;
//import com.spbsu.commons.math.vectors.VecTools;
//import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
//import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
//import com.spbsu.commons.random.FastRandom;
//import com.spbsu.commons.util.ArrayTools;
//import com.spbsu.ml.BFGrid;
//import com.spbsu.ml.ComputeDistributions;
//import com.spbsu.ml.data.impl.DistributionsDataSet;
//import com.spbsu.ml.data.set.VecDataSet;
//import com.spbsu.ml.data.stats.FeatureDistribution;
//import com.spbsu.ml.loss.L2;
//import com.spbsu.ml.loss.StatBasedLoss;
//import com.spbsu.ml.loss.WeightedLoss;
//import com.spbsu.ml.methods.VecOptimization;
//import com.spbsu.ml.models.LinearRegion;
//import com.spbsu.ml.models.smoothRegions.DistributionFunc;
//import com.spbsu.ml.models.smoothRegions.ParametricDistributionFunc;
//import com.spbsu.ml.models.smoothRegions.SmoothLinearRegion;
//import gnu.trove.list.array.TDoubleArrayList;
//
//import java.lang.reflect.Constructor;
//import java.lang.reflect.InvocationTargetException;
//import java.util.List;
//
//import static com.spbsu.commons.math.MathTools.sqr;
//
///**
// * Created by noxoomo on 10/09/2017.
// */
//public class GreedyTDSmoothLinearRegion<Loss extends StatBasedLoss> extends VecOptimization.CoordinateProjectionStub<Loss> {
//  private final int depth;
//  private final FastRandom random = new FastRandom();
//
//
//  public GreedyTDSmoothLinearRegion(final int depth) {
//    this.depth = depth;
//  }
//
//  class SmoothRegionStatistics {
//    private List<ParametricDistributionFunc> distributions;
//    private List<Vec> probs;
//    private boolean[] masks;
//    private int[] featureIds;
//    private Vec betas;
//    private double score;
//    private boolean isScoreCalculated = false;
//    private DistributionsDataSet dataSet;
//    private Vec sums;
//    private Vec weights;
//
//    private Mx cov;
//    private Mx invCov;
//    private double alpha = 0.001;
//
//
//    double score() {
//      if (!isScoreCalculated) {
//        final Vec projectedTarget = projectedTarget();
//        final double targetBetasProd = VecTools.multiply(projectedTarget, betas());
//        final Vec tmp = MxTools.multiply(inverseCovarianceMatrix(), projectedTarget);
//        final double targetThroughInvSigmaDot = VecTools.multiply(projectedTarget, tmp);
//        score = (0.5 * targetBetasProd - targetThroughInvSigmaDot);
//        isScoreCalculated = true;
//      }
//      return score;
//    }
//
//    Vec betas() {
//      if (betas == null) {
//        Vec projectedTarget = projectedTarget();
//        betas = MxTools.multiply(invCov, projectedTarget);
//      }
//      return betas;
//    }
//
//    Vec probs(int depth) {
//      if (probs.get(depth) == null) {
//        final Vec current;
//
//        int startDepth = depth - 1;
//        while (startDepth >= 0 && probs.get(startDepth) == null) {
//          --startDepth;
//        }
//        current = VecTools.copy(probs.get(startDepth));
//        for (int i = startDepth + 1; i <= depth; ++i) {
//          updateProbs(current, i);
//          probs.set(i, VecTools.copy(current));
//        }
//      }
//      return probs.get(depth);
//    }
//
//    private void updateProbs(final Vec current,
//                             final int depth) {
//      final ParametricDistributionFunc distributionFunc = distributions.get(depth);
//      final Vec feature = dataSet.feature(featureIds[depth]);
//      for (int i = 0; i < current.dim(); ++i) {
//        final double p = distributionFunc.value(feature.get(i));
//        current.set(i, current.get(i) * (masks[depth] ? 1.0 - p : p));
//      }
//    }
//
//
//    private Vec projectedTarget() {
//      return sums;
//    }
//
//    private Mx inverseCovarianceMatrix() {
//      if (invCov == null) {
//        final Mx cov = VecTools.copy(covarianceMatrix());
//        for (int i = 1; i < cov.rows(); ++i) {
//          cov.adjust(i, i, alpha);
//        }
//        invCov = MxTools.inverseCholesky(cov);
//      }
//      return invCov;
//    }
//
//    private Mx covarianceMatrix() {
//      if (cov == null) {
//        cov = new VecBasedMx(weights.dim(), weights.dim());
//        for (int i = 0; i < cov.rows(); ++i) {
//          for (int j = 0; j < cov.rows(); ++j) {
//            int idx = j < i ? i : j;
//            cov.set(i, j, weights.get(idx));
//          }
//        }
//      }
//      return cov;
//    }
//  }
//
//
//  class DistributionParamEstimator {
//    private final Vec target;
//    private final Vec weights;
//    private final SmoothRegionStatistics currentRegion;
//    private final double l2 = 1.0;
//
//
//
//    ParametricDistributionFunc nextPoint(final double bias, final double mu,
//                                         final int featureId,
//                                         final DistributionsDataSet dataSet) throws NoSuchMethodException, IllegalAccessException, InvocationTargetException, InstantiationException {
//
//      final Vec feature = dataSet.feature(featureId);
//
//      Vec gradient = new ArrayVec();
//      Mx hessian = new VecBasedMx(currentDistribution.paramDim(), currentDistribution.paramDim());
//
//      for (int doc = 0; doc < feature.dim(); ++doc) {
//        final double point = feature.get(doc);
//        final double p = currentDistribution.value(point);
//
//        final double gradMult = (target.get(doc) - weights.get(doc) * (bias + mu * p)) * mu;
//        final Vec localGrad = VecTools.scale(currentDistribution.paramGradient(point), gradMult);
//
//        Vec localHessian = currentDistribution.paramHessian(point);
//        final Vec tmp =  VecTools.outer(localGrad, localGrad);
//        localHessian = VecTools.scale(localHessian, p);
//        localHessian = VecTools.append(localHessian, tmp);
//        final double hessianMult = -weights.get(doc) * sqr(mu);
//        localHessian = VecTools.scale(localHessian, hessianMult);
//
//        gradient = VecTools.append(gradient, localGrad);
//        hessian = VecTools.append(hessian, localHessian);
//      }
//
//      hessian = VecTools.append(hessian, MxTools.diag(gradient.dim(), l2));
//      final Mx invHessian = MxTools.inverseCholesky(hessian);
//      Vec param = VecTools.subtract(currentDistribution.param(), MxTools.multiply(invHessian, gradient));
//      Constructor<?> constructor = currentDistribution.getClass().getConstructor(Vec.class);
//      return (ParametricDistributionFunc) constructor.newInstance(param);
//    }
//  }
//
//
//
//  class RegionSearcherHelper {
//    private final Vec target;
//    private final Vec weights;
//
//    RegionSearcherHelper(final Vec target,
//                         final Vec weights) {
//      this.target = target;
//      this.weights = weights;
//    }
//
//
//    EstimationResult visit(final FeatureDistribution smoothingDistribution,
//                           final DistributionsDataSet dataSet) {
//      final int featureId = smoothingDistribution.featureId();
//      final Vec feature = dataSet.feature(featureId);
//      final SufficientStatistic stat = new SufficientStatistic();
//      final DistributionFunc cumFunc = smoothingDistribution.cumulativeDistributionFunc();
//      for (int doc = 0; doc < feature.dim(); ++doc) {
//        final double p = cumFunc.value(feature.get(doc));
//        stat.add(target.get(doc), weights.get(doc), p);
//      }
//      return new EstimationResult(featureId, stat.score(), stat.bias(), stat.mu());
//    }
//
//    class EstimationResult {
//      final int featureId;
//      final double score;
//      final double bias;
//      final double mu;
//
//      EstimationResult(int featureId,
//                       double score,
//                       double bias,
//                       double mu) {
//
//        this.featureId = featureId;
//        this.score = score;
//        this.bias = bias;
//        this.mu = mu;
//      }
//    }
//  }
//
//
//  @Override
//  public SmoothLinearRegion fit(final VecDataSet learn,
//                                final Loss loss) {
//
//    final DistributionsDataSet dataSet = learn.cache().cache(ComputeDistributions.class, VecDataSet.class).compute(learn).value();
//    double currentScore = Double.POSITIVE_INFINITY;
//
//
//    final double[] scores = new double[dataSet.featureCount()];
//
//    final double[] currentWeights = new double[points.length];
//    ArrayTools.fill(currentWeights, 1.0);
//
//    TDoubleArrayList sums = new TDoubleArrayList(depth);
//    TDoubleArrayList weights = new TDoubleArrayList(depth);
//
//
//    final BFOptimizationRegion current = new BFOptimizationRegion(bds, loss, points);
//    {
//      AdditiveStatistics statistics = current.total();
//      sums.add(sum(statistics));
//      final double totalWeight = weight(statistics);
//      weights.add(totalWeight);
//    }
//
//
////
//    for (int level = 0; level < depth; ++level) {
//      current.visitAllSplits((bf, left, right) -> {
//        if (usedBF[bf.bfIndex]) {
//          scores[bf.bfIndex] = Double.POSITIVE_INFINITY;
//          currentDistributions[bf.bfIndex] = null;
//        }
//        else {
//          final double leftScore;
//
//          Vec leftBetas;
//
//          final double leftWeight = weight(left);
//          final double rightWeight = weight(right);
//          final double minExcluded = Math.min(leftWeight, rightWeight);
//
//
//          {
//            if (minExcluded > 3) {
//              final Vec regularizer = makeRegularizer(weights, leftWeight);
//              Mx invCov = makeInvMatrix(weights, leftWeight, regularizer);
//              Vec target = makeVector(sums, sum(left));
//              Vec adjustTarget = adjustTarget(target, weights, leftWeight);
//              leftBetas = MxTools.multiply(invCov, adjustTarget);
//              leftScore = calcScore(invCov, target, leftBetas);
//            }
//            else {
//              leftBetas = null;
//              leftScore = Double.POSITIVE_INFINITY;
//            }
//          }
//
//          Vec rightBetas;
//          final double rightScore;
//          {
//            if (minExcluded > 3) {
//              final Vec regularizer = makeRegularizer(weights, rightWeight);
//              Mx invCov = makeInvMatrix(weights, rightWeight, regularizer);
//              Vec target = makeVector(sums, sum(right));
//              Vec adjustTarget = adjustTarget(target, weights, rightWeight);
//              rightBetas = MxTools.multiply(invCov, adjustTarget);
//              rightScore = calcScore(invCov, target, rightBetas);
//            }
//            else {
//              rightBetas = null;
//              rightScore = Double.POSITIVE_INFINITY;
//            }
//          }
//          scores[bf.bfIndex] = leftScore > rightScore ? rightScore : leftScore;
//          isRight[bf.bfIndex] = leftScore > rightScore;
//          currentDistributions[bf.bfIndex] = leftScore > rightScore ? rightBetas : leftBetas;
//        }
//      });
//
//      final int bestSplit = ArrayTools.min(scores);
//      if (bestSplit < 0)
//        break;
//
//
//      if ((scores[bestSplit] >= currentScore))
//        break;
//
//      final BFGrid.BinaryFeature bestSplitBF = grid.bf(bestSplit);
//      final boolean bestSplitMask = isRight[bestSplitBF.bfIndex];
//
//
//      conditions.add(bestSplitBF);
//      usedBF[bestSplitBF.bfIndex] = true;
//      mask.add(bestSplitMask);
//      bestSolution = currentDistributions[bestSplitBF.bfIndex];
//      currentScore = scores[bestSplit];
//      if (level < (depth - 1)) {
//        current.split(bestSplitBF, bestSplitMask);
//
//        final AdditiveStatistics total = current.total();
//        sums.add(sum(total));
//        final double weight = weight(total);
//        weights.add(weight);
//      }
//    }
//
//    final boolean[] masks = new boolean[conditions.size()];
//    for (int i = 0; i < masks.length; i++) {
//      masks[i] = mask.get(i);
//    }
//
////
//    final double bias = bestSolution.get(0);
//    final double[] values = new double[bestSolution.dim() - 1];
//    for (int i = 0; i < values.length; ++i) {
//      values[i] = bestSolution.get(i + 1);
//    }
//
//    return new LinearRegion(conditions, masks, bias, values);
//  }
//
//  private Vec adjustTarget(Vec target, TDoubleArrayList weights, double weight) {
//    final Vec adjusted = VecTools.copy(target);
//    for (int i = 0; i < target.dim(); ++i) {
//      final double w = i < weights.size() ? weights.get(i) : weight;
////      adjusted.set(i, target.get(i) * w / (w + 1));
//      adjusted.set(i, target.get(i) * (w - 1) / w);
//    }
//    return adjusted;
//  }
//
//
//  private double calcScore(final Mx sigma, final Vec targetProj, final Vec betas) {
//    final double targetBetasProd = VecTools.multiply(targetProj, betas);
//    final Vec tmp = MxTools.multiply(sigma, targetProj);
//    final double targetThroughInvSigmaDot = VecTools.multiply(targetProj, tmp);
////    final double rss = sum2 - 2 * targetThroughInvSigmaDot + targetBetasProd;
//    return (0.5 * targetBetasProd - targetThroughInvSigmaDot);
////     return n * Math.log(rss / (n - targetProj.dim())) + betas.dim() * Math.log(n);
//  }
//
//  private Vec makeVector(TDoubleArrayList sums, double sum) {
//    Vec result = new ArrayVec(sums.size() + 1);
//    for (int i = 0; i < sums.size(); ++i) {
//      result.set(i, sums.get(i));
//    }
//    result.set(sums.size(), sum);
//    return result;
//  }
//
//  private Mx makeMatrix(TDoubleArrayList weights, double weight) {
//    final Mx cov = new VecBasedMx(weights.size() + 1, weights.size() + 1);
//    final int n = weights.size() + 1;
//    for (int i = 0; i < n; ++i) {
//      for (int j = 0; j < n; ++j) {
//        int idx = j < i ? i : j;
//        cov.set(i, j, (idx < weights.size() ? weights.get(idx) : weight));
//      }
//    }
//    return cov;
//  }
//
//  private Mx makeInvMatrix(TDoubleArrayList weights, double weight, Vec regularizer) {
//    final Mx cov = new VecBasedMx(weights.size() + 1, weights.size() + 1);
//    final int n = weights.size() + 1;
//    for (int i = 0; i < n; ++i) {
//      for (int j = 0; j < n; ++j) {
//        int idx = j < i ? i : j;
//        cov.set(i, j, (idx < weights.size() ? weights.get(idx) : weight));
//      }
//    }
//    if (regularizer != null) {
//      for (int i = 0; i < n; ++i) {
//        cov.adjust(i, i, regularizer.get(i));
//      }
//    }
//    return MxTools.inverseCholesky(cov);
//  }
//
//
//
//
//  private double weight(final AdditiveStatistics stat) {
//    if (stat instanceof L2.MSEStats) {
//      return ((L2.MSEStats) stat).weight;
//    }
//    else if (stat instanceof WeightedLoss.Stat) {
//      return weight(((WeightedLoss.Stat) stat).inside);
//    }
//    else {
//      throw new RuntimeException("error");
//    }
//  }
//
//  private double sum(final AdditiveStatistics stat) {
//    if (stat instanceof L2.MSEStats) {
//      return ((L2.MSEStats) stat).sum;
//    }
//    else if (stat instanceof WeightedLoss.Stat) {
//      return sum(((WeightedLoss.Stat) stat).inside);
//    }
//    else {
//      throw new RuntimeException("error");
//    }
//  }
//}
