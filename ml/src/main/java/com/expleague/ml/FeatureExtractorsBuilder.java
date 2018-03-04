package com.expleague.ml;

import com.expleague.commons.func.Factory;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.util.Pair;
import com.expleague.ml.data.ctrs.Ctr;
import com.expleague.ml.data.ctrs.CtrEstimationPolicy;
import com.expleague.ml.data.ctrs.CtrTarget;
import com.expleague.ml.data.ctrs.EstimationAwareCtr;
import com.expleague.ml.data.perfectHash.PerfectHash;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.CatboostPool;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.bayesianUpdaters.BetaBinomialUpdater;
import com.expleague.ml.distributions.parametric.BetaDistribution;
import com.expleague.ml.distributions.parametric.impl.BetaDistributionImpl;
import com.expleague.ml.distributions.parametric.impl.BetaVecDistributionImpl;
import com.expleague.ml.randomnessAware.DeterministicFeatureExtractor;
import com.expleague.ml.randomnessAware.VecRandomFeatureExtractor;
import gnu.trove.set.hash.TDoubleHashSet;
import org.apache.commons.math3.special.Gamma;

import java.util.*;

/**
 * Created by noxoomo on 06/11/2017.
 */
public class FeatureExtractorsBuilder {
  private VecDataSet dataSet;
  private Map<Integer, Integer> catFeatures = new TreeMap<>();
  private List<VecRandomFeatureExtractor> extractors = new ArrayList<>();
  private ArrayPermutation ctrEstimationOrder;
  private CtrEstimationPolicy policy = CtrEstimationPolicy.Greedy;
  private double priorStrength = 1.0;

  public ArrayPermutation ctrEstimationOrder() {
    return ctrEstimationOrder;
  }

  public CtrEstimationPolicy ctrEstimationPolicy() {
    return policy;
  }

  public FeatureExtractorsBuilder(Pool<?> pool) {
    this.dataSet = pool.vecData();
    final Mx data = dataSet.data();

    if (pool instanceof CatboostPool) {
      for (Integer catFeature : ((CatboostPool) pool).catFeatureIds()) {
        int uniqueValues = GridTools.uniqueValues(data.col(catFeature));
        if (uniqueValues > 1) {
          catFeatures.put(catFeature, uniqueValues);
        }
      }
    }

    final int featureCount = data.columns();
    for (int i = 0; i < featureCount; ++i) {
      if (catFeatures.containsKey(i)) {
        continue;
      }
      else {
        if (notTrivial(data.col(i))) {
          extractors.add(new DeterministicFeatureExtractor(featureCount, i));
        }
      }
    }
  }

  public FeatureExtractorsBuilder setPriorStrength(double strength) {
    this.priorStrength = strength;
    return this;
  }

  public FeatureExtractorsBuilder useRandomPermutation(final FastRandom random) {
    final int[] indices = ArrayTools.sequence(0, dataSet.length());
    ArrayTools.shuffle(indices, random);
    ctrEstimationOrder = new ArrayPermutation(indices);
    return this;
  }

  public FeatureExtractorsBuilder useNativeTime() {
    policy = CtrEstimationPolicy.TimeBased;
    ctrEstimationOrder = null;
    return this;
  }

  public List<PerfectHash<Vec>> hashes() {
    final ArrayList<PerfectHash<Vec>> perfectHashes = new ArrayList<>();
    for (Integer feature : catFeatures.keySet()) {
      perfectHashes.add(getCatFeatureHash(feature));
    }
    return perfectHashes;
  }

  public FeatureExtractorsBuilder addCtrs(final CtrTarget target) {
    for (final Integer catFeature : catFeatures.keySet()) {
      final List<Ctr<?>> ctrs = createBetaCtr(target, catFeature);
      for (Ctr<?> ctr : ctrs) {
        dataSet.cache().cache(EstimationAwareCtr.class, VecDataSet.class).estimate(policy, ctr, target.target(), ctrEstimationOrder);
        extractors.add(ctr);
      }
    }
    return this;
  }



  public FeatureExtractorsBuilder useOneHots(int limit) {
    final Mx data = dataSet.data();
    for (Map.Entry<Integer, Integer> entry : catFeatures.entrySet()) {
      final int uniqueValues = entry.getValue();
      if (uniqueValues < limit) {
        final Integer catFeature = entry.getKey();
        final DeterministicFeatureExtractor catFeatureExtractor = new DeterministicFeatureExtractor(data.columns(), catFeature);
        extractors.add(catFeatureExtractor);
        OneHotFeaturesSet.add(catFeatureExtractor);
      }
    }
    return this;
  }

  public List<VecRandomFeatureExtractor> build() {
    return extractors;
  }

  private PerfectHash<Vec> getCatFeatureHash(int featureId) {
    return dataSet.cache().cache(ComputeCatFeaturesPerfectHash.class, VecDataSet.class).hash(featureId);
  }

//  private List<RandomVariable> defaultPriors(CtrTarget target) {
//    switch (target.type()) {
//      case Binomial: {
//        final double mean = VecTools.sum(target.target()) / target.target().dim();
//        final ArrayList<RandomVariable> result = new ArrayList<>();
////        result.add(new BetaDistributionImpl(0.0, 1.0));
////        result.add(new BetaDistributionImpl(1.0, 0.0));
//        result.add(new BetaDistributionImpl(mean * priorStrength, (1.0 - mean) * priorStrength));
//        return result;
//      }
//      case Normal:
//      default: {
//        throw new RuntimeException("Unimplemented ctr target " + target.type());
//      }
//    }
//  }

  private List<Ctr<?>> createBetaCtr(final CtrTarget target,
                                     int featureId) {
    if (target.type() != CtrTarget.CtrTargetType.Binomial) {
      throw new RuntimeException();
    }
//    final List<RandomVariable<?>> priors = defaultPriors(target);
    final List<Ctr<?>> ctrs = new ArrayList<>();
//    for (final RandomVariable prior : priors) {
      final PerfectHash<Vec> catFeatureHash = getCatFeatureHash(featureId);

    final Factory betaListFactory = () -> new BetaVecDistributionImpl.BetaDistributionList();
    ctrs.add(new Ctr(betaListFactory,
                     catFeatureHash,
                     estimatePrior(target, catFeatureHash),
                     new BetaBinomialUpdater(),
                     dataSet.data().columns()));
//    }
    return ctrs;
  }

  private RandomVariable estimatePrior(final CtrTarget target,
                                       final PerfectHash<Vec> catFeatureHash) {
    switch (target.type()) {
      case Binomial: {
        return estimateBetaPrior(target.target(), catFeatureHash);
      }
      case Normal:
      default: {
        throw new RuntimeException("Unimplemented ctr target " + target.type());
      }
    }
  }

  private BetaDistribution estimateBetaPrior(final Vec target, final PerfectHash<Vec> catFeatureHash) {
    double[] positiveCounts = new double[catFeatureHash.size()];
    double[] counts = new double[catFeatureHash.size()];



    double alpha;
    double beta;

    {
      double mean = 0;
      Mx data = dataSet.data();
      for (int i = 0; i < data.rows(); ++i) {
        final int idx = catFeatureHash.id(data.row(i));
        positiveCounts[idx] += target.get(i);
        counts[idx]++;
        mean += target.get(i);
      }
//      alpha =0.5;// mean / dataSet.length();
      alpha  = mean / dataSet.length();
      beta = 1.0 - alpha;
//      alpha *= positiveCounts.length;
//      beta *= positiveCounts.length;
    }

    for (int i = 0; i < 20; ++i) {
      System.out.println("Point (" + alpha +", " + beta + "), Current likelihood: " + likelihood(positiveCounts, counts, alpha, beta));
      final Pair<Vec, Mx> ders = derAndDer2(positiveCounts, counts, alpha, beta);
      Vec direction = MxTools.multiply(MxTools.inverse(ders.second), ders.first);
      double step = 1.0;
      double newAlpha = alpha - step * direction.get(0);
      double newBeta = beta - step * direction.get(1);
      if (alpha < 1e-9) {
        alpha += 1e-9;
      }
      if (beta < 1e-9) {
        beta += 1e-9;
      }
      while (newAlpha < 1e-9 || newBeta < 1e-9) {
        step *= 0.5;
        newAlpha = alpha - step * direction.get(0);
        newBeta = beta - step * direction.get(1);
      }
      alpha = newAlpha;
      beta = newBeta;
    }
    return new BetaDistributionImpl(alpha, beta);
//    return new BetaDistributionImpl(0.5, 0.5);
  }

  private double likelihood(double[] positiveCounts, double[] counts, double alpha, double beta) {
    double ll = 0;

    for (int i = 0; i < positiveCounts.length; ++i) {
      final double first = positiveCounts[i];
      final double n = counts[i];
      final double second = n - first;
      ll += Gamma.logGamma(n + 1)  - Gamma.logGamma(first + 1) - Gamma.logGamma(second + 1);
      ll += Gamma.logGamma(first + alpha) + Gamma.logGamma(second + beta) - Gamma.logGamma(n + alpha + beta);
    }
    ll += Gamma.logGamma(alpha + beta) * positiveCounts.length;
    ll -= (Gamma.logGamma(alpha) + Gamma.logGamma(beta)) * positiveCounts.length;
    return ll;
  }

  private Pair<Vec, Mx> derAndDer2(double[] positiveCounts, double[] counts, double alpha, double beta) {
    final double lambda = 0.1;

    final Vec der = new ArrayVec(2);
    final Mx der2 = new VecBasedMx(2, 2);

    double der2Alpha= 0;
    double der2Beta= 0;
    double der2AlphaBeta= 0;

    final int k = positiveCounts.length;
    for (int i = 0; i < k; ++i) {
      final double first = positiveCounts[i];
      final double n = counts[i];
      final double second = n - first;

      der.adjust(0, Gamma.digamma(first + alpha));

      {
        final double tmp = Gamma.trigamma(first + alpha);
        der2Alpha += tmp;
      }

      der.adjust(1, Gamma.digamma(second + beta));
      {
        final double tmp = Gamma.trigamma(second + beta);
        der2Beta += tmp;
      }

      {
        final double tmp = Gamma.digamma(n + alpha + beta);
        der.adjust(0, -tmp);
        der.adjust(1, -tmp);
      }
      {
        final double tmp = Gamma.trigamma(n + alpha + beta);
        der2Alpha -= tmp;
        der2Beta -= tmp;
        der2AlphaBeta -= tmp;
      }
    }
    {
      final double tmp = Gamma.digamma(alpha + beta);

      der.adjust(0, (tmp - Gamma.digamma(alpha)) * k);
      der.adjust(1, (tmp - Gamma.digamma(beta)) * k);
    }
    {
      final double tmp = Gamma.trigamma(alpha + beta) * k;
      der2AlphaBeta += tmp;
      der2Alpha += tmp - Gamma.trigamma(alpha) * k;
      der2Beta += tmp - Gamma.trigamma(beta) * k;
    }
    der2.set(0, 0, der2Alpha + lambda);
    der2.set(1, 1, der2Beta + lambda);
    der2.set(0, 1, der2AlphaBeta);
    der2.set(1, 0, der2AlphaBeta);
//
//    der2.set(0, 0, -1.0);
//    der2.set(1, 1, -1.0);
//    der2.set(0, 1, 0);
//    der2.set(1, 0, 0);
    if ((der2Alpha  - lambda) * (der2Beta - lambda) - der2AlphaBeta * der2AlphaBeta < 0) {
      throw new RuntimeException("error: det should be positive");
    }
    return new Pair<>(der, der2);
  }


//  public static ConjugateBayesianEstimator<?> createEstimator(CtrTarget.CtrTargetType target) {
//    switch (target) {
//      case Binomial: {
//        return new BetaConjugateBayesianEstimator();
//      }
//      case Normal:
//        return new NormalConjugateBayesianEstimator();
//      default: {
//        throw new RuntimeException("Unimplemented ctr target " + target);
//      }
//    }
//  }
//
//  public static DynamicRandomVec<?> createDynamicVec(CtrTarget.CtrTargetType target) {
//    switch (target) {
//      case Binomial: {
//        return new BetaVecDistributionImpl();
//      }
//      case Normal:
//        return new NormalGammaVecDistributionImpl();
//      default: {
//        throw new RuntimeException("Unimplemented ctr target " + target);
//      }
//    }
//  }

  private boolean notTrivial(final Vec col) {
    TDoubleHashSet uniqueValues = new TDoubleHashSet();
    for (int i = 0; i < col.dim(); ++i) {
      uniqueValues.add(col.get(i));
      if (uniqueValues.size() > 1) {
        return true;
      }
    }
    return false;
  }

  public void setEstimationPolicy(final CtrEstimationPolicy estimationPolicy) {
    this.policy = estimationPolicy;
  }


}
