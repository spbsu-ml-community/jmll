package com.expleague.ml.func;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.randomnessAware.ProcessRandomnessPolicy;
import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.randomnessAware.RandomnessAwareTrans;

import java.util.Arrays;
import java.util.List;

/**
 * User: noxoomo
 */
public class RandomnessAwareEnsemble<U extends ProcessRandomnessPolicy, F extends RandomnessAwareTrans<U>> extends Ensemble<F> implements RandomnessAwareTrans<U> {
  public RandomnessAwareEnsemble(final F[] models,
                                 final Vec weights,
                                 final FastRandom random) {
    super(models, weights);
    setRandom(random);
  }

  public RandomnessAwareEnsemble(final List<F> weakModels,
                                 final double step,
                                 final FastRandom random
                                 ) {
    this(ArrayTools.toArray(weakModels), VecTools.fill(new ArrayVec(weakModels.size()), step), random);
  }

  public F last() {
    return models[size() - 1];
  }

  public int size() {
    return models.length;
  }

  public double wlast() {
    return weights.get(size() - 1);
  }

  @Override
  public int xdim() {
    return models[0].xdim();// * models.length;
  }

  @Override
  public int ydim() {
    return models[ArrayTools.max(models, RandomnessAwareTrans::ydim)].ydim();
  }

  public Class<? extends RandomnessAwareTrans> componentType() {
    return models.length > 0 ? models[0].getClass() : RandomnessAwareTrans.class;
  }

  @Override
  public boolean equals(final Object o) {
    if (this == o) return true;
    if (!(o instanceof RandomnessAwareEnsemble)) return false;

    final RandomnessAwareEnsemble ensemble = (RandomnessAwareEnsemble) o;

    return Arrays.equals(models, ensemble.models) && weights.equals(ensemble.weights);
  }

  @Override
  public int hashCode() {
    int result = Arrays.hashCode(models);
    result = 31 * result + weights.hashCode();
    return result;
  }

  @Override
  public U activePolicy() {
    return models[0].activePolicy();
  }

  @Override
  public Mx transAll(final VecDataSet dataSet) {
    Mx result = new VecBasedMx(dataSet.length(), 1);
    for (F model : models) {
      VecTools.append(result, model.transAll(dataSet));
    }
    return result;
  }

  @Override
  public RandomnessAwareTrans<U> changePolicy(final U policy) {
    for (F model : models) {
      model.changePolicy(policy);
    }
    return this;
  }

  @Override
  public void setRandom(final FastRandom random) {
    for (F model : models) {
      model.setRandom(random);
    }
  }

  @Override
  public FastRandom random() {
    return models[0].random();
  }
}
