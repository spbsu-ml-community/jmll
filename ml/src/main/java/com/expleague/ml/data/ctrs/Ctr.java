package com.expleague.ml.data.ctrs;

import com.expleague.commons.func.Action;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.bayesianEstimation.ConjugateBayesianEstimator;
import com.expleague.ml.data.perfectHash.PerfectHash;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.distributions.DynamicRandomVec;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.distributions.RandomVecBuilder;
import com.expleague.ml.distributions.samplers.RandomVecSampler;
import com.expleague.ml.randomnessAware.HashedRandomFeatureExtractor;

import java.util.function.Consumer;


/**
 * Created by noxoomo on 27/10/2017.
 */
public class Ctr<U extends RandomVariable<U>> implements HashedRandomFeatureExtractor<U> {
  private final DynamicRandomVec<U> knownCtrsTable;
  private final PerfectHash<Vec> perfectHash;
  private final ConjugateBayesianEstimator<U> estimator;
  private final U prior;
  private final int dim;

  public Ctr(final DynamicRandomVec<U> knownCtrsTable,
             final PerfectHash<Vec> perfectHash,
             final U priorDistribution,
             final ConjugateBayesianEstimator<U> estimator,
             final int dim) {
    this.knownCtrsTable = knownCtrsTable;
    this.perfectHash = perfectHash;
    this.estimator = estimator;
    this.dim = dim;
    this.prior = estimator.clone(priorDistribution);
    final Action<U> addCtr = knownCtrsTable.updater();

    for (int key = knownCtrsTable.dim(); key < perfectHash.size(); ++key) {
      addCtr.invoke(prior);
    }
    perfectHash.addListener(key -> addCtr.invoke(prior));
  }

  public int hash(final Vec featuresVec) {
    return perfectHash.id(featuresVec);
  }

  @Override
  public U variable(final int idx) {
    return idx >= 0 ? knownCtrsTable.randomVariable(idx) : prior;
  }

  @Override
  public RandomVec<U> randomVecForBins(final int[] bins) {
    return new CtrsVec(bins);
  }

  public U get(final Vec featureVec) {
    final int key = hash(featureVec);
    return knownCtrsTable.randomVariable(key);
  }

  public Ctr<U> update(final Vec featureVec,
                       final double target) {
    final int key = hash(featureVec);
    return update(key, target);
  }

  public Ctr<U> update(final int key,
                       final double target) {
    estimator.update(key, target, knownCtrsTable);
    return this;
  }

  public Ctr<U> consumeThenUpdate(final Vec featureVec,
                                  final double target,
                                  final Consumer<U> consumer) {
    final int key = hash(featureVec);
    consumer.accept(knownCtrsTable.randomVariable(key));
    estimator.update(key, target, knownCtrsTable);
    return this;
  }

  class CtrsVec extends RandomVec.IndependentCoordinatesDistribution<U> implements RandomVec<U> {
    final int[] bins;

    CtrsVec(final int[] bins) {
      this.bins = bins;
    }

    @Override
    public final U randomVariable(final int idx) {
      return variable(bins[idx]);
    }

    @Override
    public final RandomVecBuilder<U> builder() {
      return knownCtrsTable.builder();
    }

    @Override
    public final RandomVec<U> setRandomVariable(final int idx, final U var) {
      return setRandomVariable(bins[idx], var);
    }

    private final RandomVecSampler sampler = new RandomVecSampler() {
      @Override
      public final double instance(final FastRandom random, final int i) {
        return bins[i] >= 0 ? knownCtrsTable.sampler().instance(random, bins[i]) : prior.sampler().instance(random);
      }

      @Override
      public final int dim() {
        return bins.length;
      }
    };

    @Override
    public final RandomVecSampler sampler() {
      return sampler;
    }

    @Override
    public final int dim() {
      return bins.length;
    }

    @Override
    public final double expectation(final int idx) {
      return bins[idx] >= 0 ? knownCtrsTable.expectation(bins[idx]) : prior.expectation();
    }

    @Override
    public final double cumulativeProbability(final int idx, final double x) {
      return bins[idx] >= 0 ? knownCtrsTable.cumulativeProbability(bins[idx], x) : prior.cdf(x);
    }
  }

  @Override
  public final RandomVec<U> apply(final VecDataSet dataSet) {
    return dataSet.cache().cache(EstimationAwareCtr.class, VecDataSet.class).apply(this);
  }

  @Override
  public final int dim() {
    return dim;
  }

  @Override
  public final RandomVecBuilder<U> randomVecBuilder() {
    return prior.vecBuilder();
  }

  public final RandomVec<U> applyAll(final VecDataSet dataSet) {
    return dataSet.cache().cache(EstimationAwareCtr.class, VecDataSet.class).apply(this);
  }
}






