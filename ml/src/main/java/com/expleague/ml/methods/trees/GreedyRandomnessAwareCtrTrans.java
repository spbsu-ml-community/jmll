package com.expleague.ml.methods.trees;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.util.ThreadTools;
import com.expleague.ml.FeatureExtractorsBuilder;
import com.expleague.ml.bayesianEstimation.ConjugateBayesianEstimator;
import com.expleague.ml.data.ctrs.*;
import com.expleague.ml.data.perfectHash.PerfectHash;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.distributions.DynamicRandomVec;
import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.distributions.parametric.*;
import com.expleague.ml.distributions.parametric.impl.NormalGammaDistributionImpl;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.StatBasedLoss;
import com.expleague.ml.methods.RandomnessAwareVecOptimization;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.models.RandomVariableRandomnessPolicy;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

import static com.expleague.commons.math.MathTools.sqr;

/**
 * User: noxoomo
 */
public class GreedyRandomnessAwareCtrTrans<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> implements RandomnessAwareVecOptimization<Loss> {
  private final FastRandom random;
  private final List<PerfectHash<Vec>> featureExtractors;
  private CtrEstimationPolicy ctrEstimationPolicy;
  private ArrayPermutation ctrEstimationOrder;
  private RandomVariableRandomnessPolicy randomnessPolicy = RandomVariableRandomnessPolicy.Expectation;


  public GreedyRandomnessAwareCtrTrans(final List<PerfectHash<Vec>> hashes,
                                       final FastRandom random) {
    this.featureExtractors = hashes;
    this.random = random;
  }

  private static final ThreadPoolExecutor exec = ThreadTools.createBGExecutor("Linear ctr thread", -1);


  @Override
  public CtrTrans fit(final VecDataSet learn,
                      final Loss loss) {
    final double[] scores = new double[featureExtractors.size()];

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
    final NormalGammaDistribution prior = new NormalGammaDistributionImpl(priorMu, 3, 3, 3 * priorSigma2);

    final Ctr<NormalGammaDistribution>[] ctrs = new Ctr[featureExtractors.size()];
    for (int i = 0; i < ctrs.length; ++i) {
      final DynamicRandomVec<NormalGammaDistribution> dynamicVec = (DynamicRandomVec<NormalGammaDistribution>) FeatureExtractorsBuilder.createDynamicVec(CtrTarget.CtrTargetType.Normal);
      final ConjugateBayesianEstimator<NormalGammaDistribution> estimator = (ConjugateBayesianEstimator<NormalGammaDistribution>) FeatureExtractorsBuilder.createEstimator(CtrTarget.CtrTargetType.Normal);
      ctrs[i] = new Ctr<NormalGammaDistribution>(dynamicVec, featureExtractors.get(i), prior, estimator, learn.xdim());
    }
    final RandomVec<NormalGammaDistribution>[] ctrVecs = (RandomVec<NormalGammaDistribution>[]) new RandomVec[ctrs.length];
    {
      final CountDownLatch latch = new CountDownLatch(ctrs.length);
      for (int findex = 0; findex < ctrs.length; findex++) {
        final int finalFindex = findex;
        exec.execute(() -> {
          final Ctr ctr = ctrs[finalFindex];
          ctrVecs[finalFindex] = learn.cache().cache(EstimationAwareCtr.class, VecDataSet.class).estimate(ctrEstimationPolicy, ctr, loss.target(), ctrEstimationOrder);
          latch.countDown();
        });
      }
      try {
        latch.await();
      }
      catch (InterruptedException e) {
        // skip
      }
    }
    Arrays.fill(scores, 0.0);

    {
      final CountDownLatch latch = new CountDownLatch(ctrs.length);
      for (int findex = 0; findex <ctrs.length; findex++) {
        final int finalFindex = findex;
        exec.execute(() -> {
          Vec predictions;
          switch (randomnessPolicy) {
            case Sample: {
              predictions = ctrVecs[finalFindex].sampler().sample(random);
              break;
            }
            case Expectation: {
              predictions = ctrVecs[finalFindex].expectation();
              break;
            }
            default: {
              throw new RuntimeException("Unimplemented");
            }
          }
          scores[finalFindex] = VecTools.cosine(predictions, loss.target());
//          scores[finalFindex] = -VecTools.distance(predictions, loss.target());
          latch.countDown();
        });
      }
      try {
        latch.await();
      } catch (InterruptedException e) {
        // skip
      }
    }
    final int best = ArrayTools.max(scores);
    if (scores[best] <= 0) {
      System.out.println("Warning: no positive correlated feature found");
    }
    return new CtrTrans(randomnessPolicy, ctrs[best]);
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
}
