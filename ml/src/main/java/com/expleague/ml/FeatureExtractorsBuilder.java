package com.expleague.ml;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.bayesianEstimation.ConjugateBayesianEstimator;
import com.expleague.ml.bayesianEstimation.impl.BetaConjugateBayesianEstimator;
import com.expleague.ml.bayesianEstimation.impl.NormalConjugateBayesianEstimator;
import com.expleague.ml.data.ctrs.Ctr;
import com.expleague.ml.data.ctrs.CtrEstimationPolicy;
import com.expleague.ml.data.ctrs.CtrTarget;
import com.expleague.ml.data.ctrs.EstimationAwareCtr;
import com.expleague.ml.data.perfectHash.PerfectHash;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.CatboostPool;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.distributions.DynamicRandomVec;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.parametric.impl.BetaDistributionImpl;
import com.expleague.ml.distributions.parametric.impl.BetaVecDistributionImpl;
import com.expleague.ml.distributions.parametric.impl.NormalGammaVecDistributionImpl;
import com.expleague.ml.randomnessAware.DeterministicFeatureExctractor;
import com.expleague.ml.randomnessAware.VecRandomFeatureExtractor;
import gnu.trove.set.hash.TDoubleHashSet;

import java.util.*;
import java.util.stream.Collectors;

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
          extractors.add(new DeterministicFeatureExctractor(featureCount, i));
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

  public FeatureExtractorsBuilder  useNativeTime() {
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
      final List<Ctr<?>> ctrs = createCtr(target, catFeature);
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
        final DeterministicFeatureExctractor catFeatureExtractor = new DeterministicFeatureExctractor(data.columns(), catFeature);
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

  private List<RandomVariable<?>> createPrior(CtrTarget target) {
    switch (target.type()) {
      case Binomial: {
        final double mean = VecTools.sum(target.target()) / target.target().dim();
        final ArrayList<RandomVariable<?>> result = new ArrayList<>();
//        result.add(new BetaDistributionImpl(0.0, 1.0));
//        result.add(new BetaDistributionImpl(1.0, 0.0));
        result.add(new BetaDistributionImpl(mean * priorStrength, (1.0 - mean) * priorStrength));
        return result;
      }
      case Normal:
      default: {
        throw new RuntimeException("Unimplemented ctr target " + target.type());
      }
    }
  }

  private List<Ctr<?>> createCtr(CtrTarget target, int featureId) {
    final List<RandomVariable<?>> priors = createPrior(target);
    final List<Ctr<?>> ctrs = new ArrayList<>();
    for (final RandomVariable prior : priors) {
      ctrs.add(new Ctr(createDynamicVec(target.type()), getCatFeatureHash
          (featureId), prior, createEstimator(target.type()), dataSet.data().columns()));
    }
    return ctrs;
  }


  public static ConjugateBayesianEstimator<?> createEstimator(CtrTarget.CtrTargetType target) {
    switch (target) {
      case Binomial: {
        return new BetaConjugateBayesianEstimator();
      }
      case Normal:
        return new NormalConjugateBayesianEstimator();
      default: {
        throw new RuntimeException("Unimplemented ctr target " + target);
      }
    }
  }

  public static DynamicRandomVec<?> createDynamicVec(CtrTarget.CtrTargetType target) {
    switch (target) {
      case Binomial: {
        return new BetaVecDistributionImpl();
      }
      case Normal:
        return new NormalGammaVecDistributionImpl();
      default: {
        throw new RuntimeException("Unimplemented ctr target " + target);
      }
    }
  }

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

//  public static <U extends RandomVariable<U>> Ctr<?> newCtr(final Class<U> model,
//                                                            final Seq<?> values,
//                                                            final PerfectHash<Vec> hash,
//                                                            final U prior,
//                                                            final VecDataSet ds) {
//    if (BetaDistribution.class.isAssignableFrom(model)) {
//      final DynamicRandomVec<U> betaVecDistribution = (DynamicRandomVec<U>) new BetaVecDistributionImpl();
//      final ConjugateBayesianEstimator<U> betaConjugateBayesianEstimator = (ConjugateBayesianEstimator<U>) new BetaConjugateBayesianEstimator();
//      return new Ctr<U>(betaVecDistribution, hash, prior, betaConjugateBayesianEstimator, ds.data().columns());
//    } else {
//      throw new RuntimeException("Error: unknown ctr model " + model.getSimpleName());
//    }
//  }
//
//
//  public static <U extends RandomVariable<U>> Ctr<?> newCtr(final Class<U> model,
//                                                            final Seq<?> values,
//                                                            final int featureId,
//                                                            final U prior,
//                                                            final VecDataSet ds) {
//
//    if (BetaDistribution.class.isAssignableFrom(model)) {
//      final DynamicRandomVec<U> betaVecDistribution = (DynamicRandomVec<U>) new BetaVecDistributionImpl();
//      final ConjugateBayesianEstimator<U> betaConjugateBayesianEstimator = (ConjugateBayesianEstimator<U>) new BetaConjugateBayesianEstimator();
//      return new Ctr<U>(betaVecDistribution, hash, prior, betaConjugateBayesianEstimator, ds.data().columns());
//    } else {
//      throw new RuntimeException("Error: unknown ctr model " + model.getSimpleName());
//    }
//  }
}
