package com.expleague.ml.methods.trees;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.FeatureBinarization;
import com.expleague.ml.data.BinarizedFeatureDataSet;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.parametric.DeltaFunction;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.StatBasedLoss;
import com.expleague.ml.methods.RandomnessAwareVecOptimization;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.models.BinOptimizedRandomnessPolicy;
import com.expleague.ml.models.RandomnessAwareRegion;
import com.expleague.ml.randomnessAware.VecRandomFeatureExtractor;
import org.apache.commons.math3.special.Gamma;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static com.expleague.commons.math.MathTools.sqr;

/**
 * User: noxoomo
 */
public class GreedyRandomnessAwareRegion<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> implements RandomnessAwareVecOptimization<Loss> {
  private final int maxDepth;
  private final int binarization;
  private final FastRandom random;
  private final List<VecRandomFeatureExtractor> featureExtractors;
  private final BinOptimizedRandomnessPolicy policy;
  private boolean rebuildStochasticAggregates = false;
  private boolean useBootstrap = false;
  private LeavesType leavesType = LeavesType.DeterministicMean;
  private boolean forceSampledSplit = false;

  public void useBootstrap(final boolean bootstrap) {
    useBootstrap = bootstrap;
  }

  public void forceSampledSplit(final boolean forceSampledSplit) {
    this.forceSampledSplit = forceSampledSplit;
  }

  enum LeavesType {
    BayesianMean,
    DeterministicMean,
    NormalVal
  }

  public GreedyRandomnessAwareRegion(final int depth,
                                     final List<VecRandomFeatureExtractor> featureExtractors,
                                     final int binarization,
                                     final BinOptimizedRandomnessPolicy policy,
                                     final FastRandom random) {
    this.maxDepth = depth;
    this.featureExtractors = featureExtractors;
    this.binarization = binarization;
    this.policy = policy;
    this.random = random;
  }

  @Override
  public RandomnessAwareRegion fit(final VecDataSet learn,
                                          final Loss loss) {

    List<RandomnessAwareOptimizationSubset> leaves = new ArrayList<>(1 << maxDepth);
    final List<FeatureBinarization.BinaryFeature> conditions = new ArrayList<>(maxDepth);
    final List<Boolean> masks = new ArrayList<>(maxDepth);

    double currentScore = Double.POSITIVE_INFINITY;

    final BinarizedFeatureDataSet dataSet = buildBinarizedDataSet(learn);
    final BinarizedFeatureDataSet.GridHelper gridHelper = dataSet.gridHelper();

    final double[] scoresLeft = new double[gridHelper.binFeatureCount()];
    final double[] scoresRight = new double[gridHelper.binFeatureCount()];


    final double priorSigma2;
    final double priorMu;
    {
      AdditiveStatistics stat = (AdditiveStatistics) loss.statsFactory().create();
      for (int i = 0; i < learn.length(); ++i) {
        stat.append(i, 1);
      }
      final double sum = ((L2.MSEStats) stat).sum;
      final double sum2 = ((L2.MSEStats) stat).sum2;
      final double weight = ((L2.MSEStats) stat).weight;
      priorSigma2 =(sum2 - sqr(sum) / weight) / (weight - 1);
      priorMu =  sum / (weight + 1.0);
    }

    RandomnessAwareOptimizationSubset rightSet;
    RandomnessAwareOptimizationSubset leftSet;
    {
      final double weights[] = useBootstrap ? new double[learn.length()] : null;
      if (useBootstrap) {
        for (int i = 0; i < weights.length; ++i) {
          weights[i] = random.nextGamma(1.0);
        }
      }
      final RandomnessAwareOptimizationSubset allPoints = new RandomnessAwareOptimizationSubset(dataSet, loss, ArrayTools.sequence(0, learn.length()), weights, random);

      Arrays.fill(scoresLeft, 0);
      allPoints.visitAllSplits((bf, left, right) -> {
        scoresLeft[gridHelper.binaryFeatureOffset(bf)] += loss.score(left) + loss.score(right);
//        scoresLeft[gridHelper.binaryFeatureOffset(bf)] += score(left, priorMu, priorSigma2) + score(right, priorMu, priorSigma2);
      });
      final int bestSplit = ArrayTools.min(scoresLeft);
      final FeatureBinarization.BinaryFeature bestSplitBF = gridHelper.binFeature(bestSplit);
      currentScore = scoresLeft[bestSplit];

      leftSet = allPoints;
      rightSet = allPoints.split(bestSplitBF, rebuildStochasticAggregates, forceSampledSplit);
      conditions.add(bestSplitBF);
    }

    for (int level = 1; level < maxDepth; level++) {
      Arrays.fill(scoresLeft, 0.);
      Arrays.fill(scoresRight, 0.);

      final AdditiveStatistics leftTotal = leftSet.total();
      final AdditiveStatistics rightTotal = rightSet.total();

      rightSet.visitAllSplits((bf, left, right) -> {
        scoresRight[gridHelper.binaryFeatureOffset(bf)] += loss.score(left) + loss.score(right) + loss.score(leftTotal);
//        scoresLeft[gridHelper.binaryFeatureOffset(bf)] += score(left, priorMu, priorSigma2) + score(right, priorMu, priorSigma2) + score(leftTotal, priorMu, priorSigma2);

      });

      leftSet.visitAllSplits((bf, left, right) -> {
        scoresLeft[gridHelper.binaryFeatureOffset(bf)] += loss.score(left) + loss.score(right) + loss.score(rightTotal);
//        scoresLeft[gridHelper.binaryFeatureOffset(bf)] += score(left, priorMu, priorSigma2) + score(right, priorMu, priorSigma2) + score(rightTotal, priorMu, priorSigma2);
      });



      final int bestSplitLeft = ArrayTools.min(scoresLeft);
      final int bestSplitRight = ArrayTools.min(scoresRight);

      final boolean splitRight = !(scoresLeft[bestSplitLeft] < scoresRight[bestSplitRight]);
      final double bestScore = splitRight ? scoresRight[bestSplitRight] : scoresLeft[bestSplitLeft];
      final int bestSplit = splitRight ? bestSplitRight : bestSplitLeft;

      if (bestSplit < 0 || bestScore + 1e-9 >= currentScore) {
        break;
      }

      final FeatureBinarization.BinaryFeature bestSplitBF = gridHelper.binFeature(bestSplit);
      conditions.add(bestSplitBF);

      if (splitRight) {
        leaves.add(leftSet);
        masks.add(true);
        leftSet = rightSet;
      } else {
        leaves.add(rightSet);
        masks.add(false);
      }
      rightSet = leftSet.split(bestSplitBF, rebuildStochasticAggregates, forceSampledSplit);
      currentScore = bestScore;
    }

    leaves.add(leftSet);
    leaves.add(rightSet);
    masks.add(true);

    final RandomVariable[] step = new RandomVariable[leaves.size()];

    for (int i = 0; i < step.length; i++) {
      final AdditiveStatistics total = leaves.get(i).total();
      final double sum = ((L2.MSEStats) total).sum;
      final double weight = ((L2.MSEStats) total).weight;
      final double bestInc = sum / (weight + 1);
      step[i] = (DeltaFunction) () -> bestInc;
    }
    return new RandomnessAwareRegion(conditions, masks, step);
  }

  final double lambda0 = 1.0;
  final double alpha0 = 0.5;

  private double score(final AdditiveStatistics stat, final double mu0, final double priorSigma2) {
    final double sum = ((L2.MSEStats) stat).sum;
    final double sum2 = ((L2.MSEStats) stat).sum2;
    final double weight = ((L2.MSEStats) stat).weight;

    if (weight < 3) {
      return 0;
    }

    final double beta0 = priorSigma2 * alpha0;

    final double mu = (lambda0 * mu0 + sum) / (lambda0 + weight);
    final double lambda = lambda0 + weight;
    final double alpha = alpha0 + weight / 2;
    final double beta = beta0 + 0.5 * (sum2 - sqr(sum) / weight) + weight * lambda0 * sqr(sum / weight - mu0) / (lambda0 + weight) / 2;

    double score = 0.5 * Math.log(lambda0) + alpha0 * Math.log(beta0) - (weight + 1) * 0.5 * Math.log(2 * Math.PI) - Gamma.logGamma(alpha0);
    score -= alpha * Math.log(beta) + 0.5 * Math.log(lambda) - 0.5 * Math.log(2 * Math.PI) - Gamma.logGamma(alpha);
    return -score;
//    return Gamma.logGamma(alpha) - Gamma.logGamma(alpha0) + alpha0 * Math.log(beta0) - alpha * Math.log(beta) + 0.5 * (Math.log(lambda0) - Math.log(lambda)) - 0.5 * weight * (Math.log(2 * Math.PI));
  }

  private double mean(final AdditiveStatistics stat, final double mu0) {
    final double sum = ((L2.MSEStats) stat).sum;
    final double weight = ((L2.MSEStats) stat).weight;
    return (lambda0 * mu0 + sum) / (lambda0 + weight);
  }

  private BinarizedFeatureDataSet ds = null;

  private BinarizedFeatureDataSet buildBinarizedDataSet(final VecDataSet learn) {
    if (ds == null) {
      final BinarizedFeatureDataSet.Builder builder = new BinarizedFeatureDataSet.Builder(learn, binarization, random);
      builder.setPolicy(policy);
      featureExtractors.forEach(builder::addFeature);
      ds = builder.build();
    }
    return ds;
  }

}
