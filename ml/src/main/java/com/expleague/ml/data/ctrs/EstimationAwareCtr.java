package com.expleague.ml.data.ctrs;

import com.expleague.commons.func.Computable;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.distributions.RandomList;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.distributions.RandomVecBuilder;

import java.util.WeakHashMap;

/**
 * Created by noxoomo on 29/10/2017.
 */
public class EstimationAwareCtr implements Computable<VecDataSet, EstimationAwareCtr> {
  private VecDataSet dataSet;
  private WeakHashMap<Ctr, RandomVec> computedCtrs = new WeakHashMap<>();

  public <U extends RandomVariable> RandomVec estimateTimeAware(final Ctr<U> ctr,
                                                                final Vec target,
                                                                final ArrayPermutation time) {
    if (!contains(ctr)) {
      final RandomList<U> orderedCtr = ctr.emptyList();
      for (int i = 0; i < dataSet.length(); ++i) {
        final Mx data = dataSet.data();
        final int idx = time != null ? time.forward(i) : i;
        ctr.consumeThenUpdate(data.row(idx), target.get(idx), orderedCtr::add);
      }

      if (time != null) {
        final RandomVecBuilder<U> directBuilder = ctr.emptyList();
        for (int i = 0; i < dataSet.length(); ++i) {
          final int idx = time.backward(i);
          directBuilder.add(orderedCtr.get(idx));
        }
        synchronized (this) {
          computedCtrs.put(ctr, directBuilder.build());
        }
      } else {
        synchronized (this) {
          computedCtrs.put(ctr, orderedCtr.build());
        }
      }
    }
    return get(ctr);
  }

  synchronized RandomVec get(final Ctr<?> ctr) {
    return computedCtrs.get(ctr);
  }

  synchronized boolean contains(final Ctr<?> ctr) {
    return computedCtrs.containsKey(ctr);
  }


  public RandomVec estimateGreedy(final Ctr<?> ctr,
                                  final Vec target) {
    if (!contains(ctr)) {
      int[] bins = new int[dataSet.length()];

      final Mx data = dataSet.data();
      for (int i = 0; i < dataSet.length(); ++i) {
        bins[i] = ctr.hash(data.row(i));
        ctr.update(bins[i], target.get(i));
      }
      final RandomVec result = ctr.randomVecForBins(bins);

      synchronized (this) {
        computedCtrs.put(ctr, result);
      }
    }
    return computedCtrs.get(ctr);
  }

  //just computeAll without update
  public <U extends RandomVariable> RandomVec apply(final Ctr<U> ctr) {
    if (!contains(ctr)) {
      //we don't want to cache it. RAM is limited resource and java already use it a lot :)
      int[] bins = new int[dataSet.length()];
      final Mx data = dataSet.data();
      for (int i = 0; i < dataSet.length(); ++i) {
        bins[i] = ctr.hash(data.row(i));
      }
      return ctr.randomVecForBins(bins);
    }
    //return ctr from estimation
    return computedCtrs.get(ctr);
  }

  public RandomVec estimate(final CtrEstimationPolicy policy,
                            final Ctr<?> ctr,
                            final Vec target) {
    return estimate(policy, ctr, target, null);
  }

  public RandomVec estimate(final CtrEstimationPolicy policy,
                            final Ctr<?> ctr,
                            final Vec target,
                            final ArrayPermutation permutation) {
    switch (policy) {
      case Greedy: {
        return estimateGreedy(ctr, target);
      }
      case TimeBased: {
        return estimateTimeAware(ctr, target, permutation);
      }
      default: {
        throw new RuntimeException("Unknown compute policy " + policy);
      }
    }
  }

  @Override
  public EstimationAwareCtr compute(final VecDataSet argument) {
    this.dataSet = argument;
    return this;
  }
}
