package com.expleague.ml.methods.trees;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.util.ThreadTools;
import com.expleague.ml.FeatureBinarization;
import com.expleague.ml.data.BinarizedFeatureDataSet;
import com.expleague.ml.data.ctrs.CtrEstimationPolicy;
import com.expleague.ml.data.perfectHash.PerfectHash;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.distributions.DistributionConvolution;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.distributions.StupidOnlyNormalDistributionConvolution;
import com.expleague.ml.distributions.parametric.NormalDistributionImpl;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.StatBasedLoss;
import com.expleague.ml.methods.RandomnessAwareVecOptimization;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.models.BinOptimizedRandomnessPolicy;
import com.expleague.ml.models.RandomVariableRandomnessPolicy;
import com.expleague.ml.models.RandomnessAwareObliviousTree;
import com.expleague.ml.randomnessAware.RandomFunc;
import com.expleague.ml.randomnessAware.VecRandomFeatureExtractor;
import org.apache.commons.math3.special.Gamma;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.ListIterator;
import java.util.concurrent.ThreadPoolExecutor;

import static com.expleague.commons.math.MathTools.sqr;

/**
 * User: noxoomo
 */
public class GreedyRandomnessAwareObliviousTree<Loss extends StatBasedLoss>  implements RandomnessAwareVecOptimization<Loss> {
  private final int depth;
  private final int binarization;
  private final FastRandom random;
  private final List<VecRandomFeatureExtractor> featureExtractors;
  private boolean rebuildStochasticAggregates = false;
  private boolean useBootstrap = false;
  private boolean forceSampledSplit = false;
  private BinOptimizedRandomnessPolicy policy = BinOptimizedRandomnessPolicy.SampleBin;
  private LeavesType leavesType = LeavesType.DeterministicMean;

  private List<PerfectHash> featureHashes;
  private CtrEstimationPolicy ctrEstimationPolicy;
  private ArrayPermutation ctrEstimationOrder;
  private RandomVariableRandomnessPolicy randomnessPolicy = RandomVariableRandomnessPolicy.Expectation;
  private final DistributionConvolution convolution = new StupidOnlyNormalDistributionConvolution();
  public void useBootstrap(final boolean bootstrap) {
    this.useBootstrap = bootstrap;
  }

  public GreedyRandomnessAwareObliviousTree forceSampledSplit(final boolean forceSampledSplit) {
    this.forceSampledSplit = forceSampledSplit;
    return this;
  }


  enum LeavesType {
    DeterministicMean,
    EstimationAwareMean
  }

  public GreedyRandomnessAwareObliviousTree(final int depth,
                                            final List<VecRandomFeatureExtractor> featureExtractors,
                                            final int binarization,
                                            final BinOptimizedRandomnessPolicy policy,
                                            final FastRandom random) {
    this.depth = depth;
    this.featureExtractors = featureExtractors;
    this.binarization = binarization;
    this.random = random;
    this.policy = policy;
  }

  int i = 0;

  @Override
  public RandomVec emptyVec(int dim) {
    return convolution.empty(dim);
  }

  @Override
  public RandomVariable emptyVar() {
    return convolution.empty();
  }

  @Override
  public RandomFunc fit(final VecDataSet learn,
                        final Loss loss) {
    List<RandomnessAwareOptimizationSubset> leaves = new ArrayList<>(1 << depth);
    final List<FeatureBinarization.BinaryFeature> conditions = new ArrayList<>(depth);
    double currentScore = Double.POSITIVE_INFINITY;

    final BinarizedFeatureDataSet dataSet = buildBinarizedDataSet(learn, loss);
    final BinarizedFeatureDataSet.GridHelper gridHelper = dataSet.gridHelper();
    final double[] scores = new double[gridHelper.binFeatureCount()];
    {
      final double weights[] = useBootstrap ? new double[learn.length()] : null;
      if (useBootstrap) {
        for (int i = 0; i < weights.length; ++i) {
          weights[i] = random.nextPoisson(1.0);
//          weights[i] = random.nextGamma(1.0);
        }
      }
      leaves.add(new RandomnessAwareOptimizationSubset(dataSet, loss, ArrayTools.sequence(0, learn.length()), weights, random));
    }
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
      priorSigma2 = (sum2 - sqr(sum) / weight) / (weight - 1);
      priorMu = 0;//sum / (weight + 1.0);
    }

    for (int level = 0; level < depth; level++) {
      Arrays.fill(scores, 0.);
      for (final RandomnessAwareOptimizationSubset leaf : leaves) {
//        leaf.visitAllSplits((bf, left, right) -> scores[gridHelper.binaryFeatureOffset(bf)] += score(left, priorMu, priorSigma2) + score(right, priorMu, priorSigma2));
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

    for (int i = 0; i < step.length; i++) {
      final AdditiveStatistics total = leaves.get(i).total();
      final double sum = ((L2.MSEStats) total).sum;
      final double sum2 = ((L2.MSEStats) total).sum2;
      final double weight = ((L2.MSEStats) total).weight;


      if (weight <= 1) {
        step[i] = new NormalDistributionImpl(0, 0);
      } else {
//        switch (leavesType) {
//          case BayesianMean: {
//            final double beta0 = priorSigma2 * alpha0;
//
//            final double mu = (lambda0 * priorMu + sum) / (lambda0 + weight);
//            final double lambda = lambda0 + weight;
//            final double alpha = alpha0 + weight / 2;
//            final double beta = beta0 + 0.5 * (sum2 - sqr(sum) / weight) + weight * lambda0 * sqr(sum / weight - priorMu) / (lambda0 + weight) / 2;
//
//
//            step[i] = new StudentDistribution.Impl(2 * alpha, mu, beta * (lambda + 1) / (alpha * lambda));
//            break;
//          }
//          case EstimationAwareMean: {
            final double var = (sum2 / weight - sum * sum / weight / weight) * weight * weight / (weight - 1) / (weight - 1);
//            final double sd = Math.sqrt(var / weight);
            if (var <= 0) {
              step[i] = new NormalDistributionImpl(0, 0);
            } else {
              final double sd = Math.sqrt(var);
              step[i] = new NormalDistributionImpl(sum / (weight + 1), sd);
            }
//            step[i] = new NormalDistributionImpl(sum / weight, );
//            break;
//          }
//          case DeterministicMean: {
//            final double mean = mean(total, priorMu);
//            step[i] = new NormalDistributionImpl(mean, 0);
//            break;
//          }
//          default: {
//            throw new RuntimeException("Unknown leave type");
//          }
//        }
      }
    }
    return new RandomnessAwareObliviousTree(conditions, step, convolution);
  }

  final double lambda0 = 1;
  final double alpha0 = 1;

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
    final double beta = beta0 + 0.5 * (sum2 - sqr(sum) / weight) + weight * lambda0 * sqr(sum / weight - mu0) / lambda / 2;
    final double tau = (alpha - 0.5) / beta; //mode
//    double score = 0.5 * Math.log(lambda0) + alpha0 * Math.log(beta0) - (weight + 1) * 0.5 * Math.log(2 * Math.PI) - Gamma.logGamma(alpha0);
//    score -= alpha * Math.log(beta) + 0.5 * Math.log(lambda) - 0.5 * Math.log(2 * Math.PI) - Gamma.logGamma(alpha);
//    return (beta / alpha) * weight;//-score;
//    return Gamma.logGamma(alpha) - Gamma.logGamma(alpha0) + alpha0 * Math.log(beta0) - alpha * Math.log(beta) + 0.5 * (Math.log(lambda0) - Math.log(lambda)) - 0.5 * weight * (Math.log(2 * Math.PI));

    return -(weight * 0.5 * Math.log(tau) - 0.5 * tau * (sum2 - 2 * sum * mu + weight * sqr(mu)) - 0.5 * weight * Math.log(2 * Math.PI) + 0.5 * Math.log(lambda0) + alpha0 * Math.log(beta0)
        + (alpha0 - 0.5) * Math.log(tau) - beta * tau - lambda0 * tau * 0.5 * sqr(mu - mu0) - Gamma.logGamma(alpha0) - 0.5 * Math.log(2 * Math.PI));
  }

  private double mean(final AdditiveStatistics stat, final double mu0) {
    final double sum = ((L2.MSEStats) stat).sum;
    final double weight = ((L2.MSEStats) stat).weight;
    return (lambda0 * mu0 + sum) / (lambda0 + weight);
  }

  private BinarizedFeatureDataSet ds = null;

  private BinarizedFeatureDataSet buildBinarizedDataSet(final VecDataSet learn, Loss loss) {
//    if (ds == null) {
//      final BinarizedFeatureDataSet.Builder builder = new BinarizedFeatureDataSet.Builder(learn, binarization, random);
//      builder.setPolicy(policy);
//      featureExtractors.stream().filter(extractor -> extractor instanceof DeterministicFeatureExtractor).forEach(builder::addFeature);
//      {
//        {
//          final double priorSigma2;
//          final double priorMu;
//          {
//            AdditiveStatistics stat = (AdditiveStatistics) loss.statsFactory().create();
//            for (int i = 0; i < learn.length(); ++i) {
//              stat.append(i, 1);
//            }
//            final double sum = ((L2.MSEStats) stat).sum;
//            final double sum2 = ((L2.MSEStats) stat).sum2;
//            final double weight = ((L2.MSEStats) stat).weight;
//            priorSigma2 =(sum2 - sqr(sum) / weight) / (weight - 1);
//            priorMu =  sum / (weight + 1.0);
//          }
//          final NormalGammaDistribution prior = new NormalGammaDistributionImpl(priorMu, 10, 10, 10 * priorSigma2);
//
//          final Ctr<NormalGammaDistribution>[] ctrs = new Ctr[featureHashes.size()];
//          for (int i = 0; i < ctrs.length; ++i) {
//            final DynamicRandomVec<NormalGammaDistribution> dynamicVec = (DynamicRandomVec<NormalGammaDistribution>) FeatureExtractorsBuilder.createDynamicVec(CtrTarget.CtrTargetType.Normal);
//            final ConjugateBayesianEstimator<NormalGammaDistribution> estimator = (ConjugateBayesianEstimator<NormalGammaDistribution>) FeatureExtractorsBuilder.createEstimator(CtrTarget.CtrTargetType.Normal);
//            ctrs[i] = new Ctr<NormalGammaDistribution>(dynamicVec, featureHashes.get(i), prior, estimator, learn.xdim());
//          }
//          final RandomVec<NormalGammaDistribution>[] ctrVecs = (RandomVec<NormalGammaDistribution>[]) new RandomVec[ctrs.length];
//          {
//            final CountDownLatch latch = new CountDownLatch(ctrs.length);
//            for (int findex = 0; findex < ctrs.length; findex++) {
//              final int finalFindex = findex;
//              exec.execute(() -> {
//                final Ctr ctr = ctrs[finalFindex];
//                ctrVecs[finalFindex] = learn.cache().cache(EstimationAwareCtr.class, VecDataSet.class).estimate(ctrEstimationPolicy, ctr, loss.target(), ctrEstimationOrder);
//                latch.countDown();
//              });
//            }
//            try {
//              latch.await();
//            }
//            catch (InterruptedException e) {
//              // skip
//            }
//          }
//          for (Ctr ctr : ctrs) {
//            builder.addFeature(ctr);
//          }
//        }
//      }
//      return builder.bu*/ild();
//    }
//    return ds;

    if (ds == null) {
      final BinarizedFeatureDataSet.Builder builder = new BinarizedFeatureDataSet.Builder(learn, binarization, random);
      builder.setPolicy(policy);
      featureExtractors.forEach(builder::addFeature);
      return builder.build();
    }
    return ds;
  }

  public GreedyRandomnessAwareObliviousTree setFeatureHashes(final List<PerfectHash> featureHashes) {
    this.featureHashes = featureHashes;
    return this;
  }

  public void setCtrEstimationPolicy(final CtrEstimationPolicy ctrEstimationPolicy) {
    this.ctrEstimationPolicy = ctrEstimationPolicy;
  }

  public void setCtrEstimationOrder(final ArrayPermutation ctrEstimationOrder) {
    this.ctrEstimationOrder = ctrEstimationOrder;
  }

  public void setRandomnessPolicy(final RandomVariableRandomnessPolicy randomnessPolicy) {
    this.randomnessPolicy = randomnessPolicy;
  }

  private static final ThreadPoolExecutor exec = ThreadTools.createBGExecutor("gradient ctr thread", -1);

}
