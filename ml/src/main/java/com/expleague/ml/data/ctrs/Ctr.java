package com.expleague.ml.data.ctrs;

import com.expleague.commons.func.Factory;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.data.perfectHash.PerfectHash;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.distributions.*;
import com.expleague.ml.randomnessAware.HashedRandomFeatureExtractor;

import java.util.function.Consumer;


/**
 * Created by noxoomo on 27/10/2017.
 */
public class Ctr<U extends RandomVariable>  implements HashedRandomFeatureExtractor<U> {
  private final RandomList<U> ctrs;
  private final PerfectHash<Vec> perfectHash;
  private final NumericBayesianUpdater<U, VecSufficientStat> bayesianUpdater;
  private final U prior;
  private final int dim;
  private Factory<RandomList<U>> factory;

  public Ctr(final Factory<RandomList<U>> factory,
             final PerfectHash<Vec> perfectHash,
             final U priorDistribution,
             final NumericBayesianUpdater<U, VecSufficientStat> bayesianUpdater,
             final int dim) {
    this.factory = factory;
    this.ctrs = factory.create();
    this.perfectHash = perfectHash;
    this.bayesianUpdater = bayesianUpdater;
    this.dim = dim;
    this.prior = priorDistribution;

    for (int key = 0; key < perfectHash.size(); ++key) {
      this.ctrs.add(prior);
    }
    perfectHash.addListener(key -> this.ctrs.add(prior));
  }

  public PerfectHash<Vec> ctrHash() {
    return perfectHash;
  }

  public int hash(final Vec featuresVec) {
    return perfectHash.id(featuresVec);
  }

  @Override
  public U variable(final int idx) {
    return idx >= 0 ? ctrs.get(idx) : prior;
  }

  @Override
  public RandomVec randomVecForBins(final int[] bins) {
    return new CtrsVec(bins);
  }

  public U get(final Vec featureVec) {
    final int key = hash(featureVec);
    return ctrs.get(key);
  }

  public Ctr<U> update(final Vec featureVec,
                       final double target) {
    final int key = hash(featureVec);
    return update(key, target);
  }

  public Ctr<U> update(final int key,
                       final double target) {
    final U ctr = ctrs.get(key);
    bayesianUpdater.posteriorTo(ctr, target, 1.0, ctr);
    return this;
  }

  public Ctr<U> consumeThenUpdate(final Vec featureVec,
                                  final double target,
                                  final Consumer<U> consumer) {
    final int key = hash(featureVec);
    final U ctr = ctrs.get(key);
    consumer.accept(ctr);
    bayesianUpdater.posteriorTo(ctr, target, 1.0, ctr);
    return this;
  }

  class CtrsVec extends RandomVec.CoordinateIndependentStub implements RandomVec {
    final int[] bins;

    CtrsVec(final int[] bins) {
      this.bins = bins;
    }

    @Override
    public final int length() {
      return bins.length;
    }

    @Override
    public U at(int idx) {
      return variable(bins[idx]);
    }

    @Override
    public double instance(int idx, FastRandom random) {
      return at(idx).instance(random);
    }

    @Override
    public double logDensity(int idx, double value) {
      return at(idx).logDensity(value);
    }

    @Override
    public double cdf(int idx, double value) {
      return at(idx).cdf(value);
    }
  }

  @Override
  public final RandomVec computeAll(final VecDataSet dataSet) {
    return dataSet.cache().cache(EstimationAwareCtr.class, VecDataSet.class).apply(this);
  }


  @Override
  public final int dim() {
    return dim;
  }

  public RandomList<U> emptyList() {
    return factory.create();
  }

}



