package com.expleague.ml.data.ctrs;

import com.expleague.commons.func.Computable;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.expleague.ml.data.set.VecDataSet;
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

  public <U extends RandomVariable<U>> RandomVec<U> estimateTimeAware(final Ctr<U> ctr,
                                                                      final Vec target,
                                                                      final ArrayPermutation time) {
    if (!computedCtrs.containsKey(ctr)) {
      final RandomVecBuilder<U> builder =  ctr.randomVecBuilder();
      for (int i = 0; i < dataSet.length(); ++i) {
        final Mx data = dataSet.data();
        final int idx = time != null ? time.forward(i) : i;
        ctr.consumeThenUpdate(data.row(idx), target.get(idx), builder::add);
      }
      final RandomVec<U> orderdCtr = builder.build();
      if (time != null) {
        final RandomVecBuilder<U> directBuilder = ctr.randomVecBuilder();
        for (int i = 0; i < dataSet.length(); ++i) {
          final int idx = time.backward(i);
          directBuilder.add(orderdCtr.randomVariable(idx));
        }
        computedCtrs.put(ctr, directBuilder.build());
      } else {
        computedCtrs.put(ctr, orderdCtr);
      }
    }
    return computedCtrs.get(ctr);
  }

  public <U extends RandomVariable<U>> RandomVec<U> estimateGreedy(final Ctr<U> ctr,
                                                                   final Vec target) {
    if (!computedCtrs.containsKey(ctr)) {
      int[] bins = new int[dataSet.length()];

      final Mx data = dataSet.data();
      for (int i = 0; i < dataSet.length(); ++i) {
        bins[i] = ctr.hash(data.row(i));
        ctr.update(bins[i], target.get(i));
      }
      computedCtrs.put(ctr, ctr.randomVecForBins(bins));
    }
    return computedCtrs.get(ctr);
  }

  //just apply without update
  public <U extends RandomVariable<U>> RandomVec<U> apply(final Ctr<U> ctr) {
    if (!computedCtrs.containsKey(ctr)) {
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

  public <U extends RandomVariable<U>> RandomVec<U> estimate(final CtrEstimationPolicy policy,
                                                             final Ctr<U> ctr,
                                                             final Vec target) {
    return estimate(policy, ctr, target, null);
  }

  public <U extends RandomVariable<U>> RandomVec<U> estimate(final CtrEstimationPolicy policy,
                                                             final Ctr<U> ctr,
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
