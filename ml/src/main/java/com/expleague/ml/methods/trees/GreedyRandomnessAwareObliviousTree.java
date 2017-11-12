package com.expleague.ml.methods.trees;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.FeatureBinarization;
import com.expleague.ml.data.BinarizedFeatureDataSet;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.parametric.DeltaFunction;
import com.expleague.ml.distributions.parametric.NormalDistribution;
import com.expleague.ml.distributions.parametric.StudentDistriubtion;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.StatBasedLoss;
import com.expleague.ml.methods.RandomnessAwareVecOptimization;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.models.RandomnessAwareObliviousTree;
import com.expleague.ml.randomnessAware.VecRandomFeatureExtractor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.ListIterator;

import static com.expleague.commons.math.MathTools.sqr;

/**
 * User: noxoomo
 */
public class GreedyRandomnessAwareObliviousTree<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> implements RandomnessAwareVecOptimization<Loss> {
  private final int depth;
  private final int binarization;
  private final FastRandom random;
  private final List<VecRandomFeatureExtractor> featureExtractors;
  private boolean sampled = true;
  private boolean rebuildStochasticAggregates = false;
  private boolean useBootstrap = false;
  private boolean forceSampledSplit = false;
  private LeavesType leavesType = LeavesType.DeterministicMean;

  public void useBootstrap(final boolean bootstrap) {
    this.useBootstrap = bootstrap;
  }

  public GreedyRandomnessAwareObliviousTree forceSampledSplit(final boolean forceSampledSplit) {
    this.forceSampledSplit = forceSampledSplit;
    return this;
  }

  enum LeavesType {
    BayesianMean,
    DeterministicMean,
    NormalVal
  }

  public GreedyRandomnessAwareObliviousTree(final int depth,
                                            final List<VecRandomFeatureExtractor> featureExtractors,
                                            final int binarization,
                                            final boolean sampled,
                                            final FastRandom random) {
    this.depth = depth;
    this.featureExtractors = featureExtractors;
    this.binarization = binarization;
    this.random = random;
    this.sampled = sampled;
  }

  @Override
  public RandomnessAwareObliviousTree fit(final VecDataSet learn,
                                          final Loss loss) {

    List<RandomnessAwareOptimizationSubset> leaves = new ArrayList<>(1 << depth);
    final List<FeatureBinarization.BinaryFeature> conditions = new ArrayList<>(depth);
    double currentScore = Double.POSITIVE_INFINITY;

    final BinarizedFeatureDataSet dataSet = buildBinarizedDataSet(learn);
    final BinarizedFeatureDataSet.GridHelper gridHelper = dataSet.gridHelper();
    final double[] scores = new double[gridHelper.binFeatureCount()];
    {
      final double weights[] = useBootstrap ? new double[learn.length()] : null;
      if (useBootstrap) {
        for (int i = 0; i < weights.length; ++i) {
          weights[i] = random.nextPoisson(1.0);
        }
      }
      leaves.add(new RandomnessAwareOptimizationSubset(dataSet, loss, ArrayTools.sequence(0, learn.length()), weights, random));
    }

    for (int level = 0; level < depth; level++) {
      Arrays.fill(scores, 0.);
      for (final RandomnessAwareOptimizationSubset leaf : leaves) {
        leaf.visitAllSplits((bf, left, right) -> scores[gridHelper.binaryFeatureOffset(bf)] += loss.score(left) + loss.score(right));
      }
      final int bestSplit = ArrayTools.min(scores);

      if (bestSplit < 0 || scores[bestSplit] + 1e-9 >= currentScore) {
        break;
      }

      final FeatureBinarization.BinaryFeature bestSplitBF = gridHelper.binFeature(bestSplit);
      final List<RandomnessAwareOptimizationSubset> next = new ArrayList<>(leaves.size() * 2);
      final ListIterator<RandomnessAwareOptimizationSubset> iter = leaves.listIterator();
      while (iter.hasNext()) {
        final RandomnessAwareOptimizationSubset subset = iter.next();
        next.add(subset);
        next.add(subset.split(bestSplitBF, rebuildStochasticAggregates, forceSampledSplit));
      }
      conditions.add(bestSplitBF);
      leaves = next;
      currentScore = scores[bestSplit];
    }

    final RandomVariable[] step = new RandomVariable[leaves.size()];

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
      priorSigma2 = sum2 / weight - sqr(sum / weight);
      priorMu = sum / weight;
    }

    for (int i = 0; i < step.length; i++) {
      final AdditiveStatistics total = leaves.get(i).total();
      final double sum = ((L2.MSEStats) total).sum;
      final double sum2 = ((L2.MSEStats) total).sum2;
      final double weight = ((L2.MSEStats) total).weight;


      if (weight < 2) {
        step[i] = (DeltaFunction) () -> 0;
      }

      else {
        switch (leavesType) {
          case BayesianMean: {
            final double mu0 = priorMu;
            final double v0 = 3.0;
            final double alpha0 = 3;
            final double beta0 = priorSigma2 * alpha0;


            final double mu = (v0 * mu0 + sum) / (v0 + weight);
            final double v = v0 + weight;
            final double alpha = alpha0 + weight / 2;
            final double beta = beta0 + 0.5 * (sum2 - sqr(sum) / weight) + weight * v0 * sqr(sum / weight - mu0) / (v0 + weight) / 2;

            step[i] = new StudentDistriubtion.Impl(2 * alpha, mu, beta * (v + 1) / (alpha * v));
            break;
          }
          case NormalVal: {
            step[i] = new NormalDistribution.Impl(sum / weight, 1.0 / (sum2 / weight - sqr(sum / weight)));
            break;

          }
          case DeterministicMean: {
            final double mean = sum / (weight + 1);
            step[i] = (DeltaFunction) () -> mean;
            break;
          }
          default: {
            throw new RuntimeException("Unknown leave type");
          }
        }
      }
    }
    return new RandomnessAwareObliviousTree(conditions, step);
  }

  private BinarizedFeatureDataSet ds = null;

  private BinarizedFeatureDataSet buildBinarizedDataSet(final VecDataSet learn) {
    if (ds == null) {
      final BinarizedFeatureDataSet.Builder builder = new BinarizedFeatureDataSet.Builder(learn, binarization, random);
      builder.setSampledFlag(sampled);
      featureExtractors.forEach(builder::addFeature);
      ds = builder.build();
    }
    return ds;
  }

}
