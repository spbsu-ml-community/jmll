package com.expleague.ml.func;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.distributions.VecDistribution;
import com.expleague.ml.randomnessAware.RandomFunc;

import java.util.Arrays;
import java.util.List;

/**
 * User: noxoomo
 */
public class RandomFuncEnsemble<F extends RandomFunc> implements RandomFunc {
  private F[] models;
  private Vec weights;

  public RandomFuncEnsemble(final F[] models,
                            final Vec weights) {
    this.models = models;
    this.weights = weights;
  }

  public RandomFuncEnsemble(final List<F> weakModels,
                            final double step) {
    this(ArrayTools.toArray(weakModels), VecTools.fill(new ArrayVec(weakModels.size()), step));
  }

  public F last() {
    return models[size() - 1];
  }

  public double wlast() {
    return weights.get(weights.dim() - 1);
  }

  public int size() {
    return models.length;
  }


  @Override
  public RandomVariable emptyVar() {
    return models[0].emptyVar();
  }

  @Override
  public RandomVec emptyVec(int dim) {
    return models[0].emptyVec(dim);
  }

  @Override
  public RandomVariable appendTo(final double scale,
                                final Vec vec,
                                final RandomVariable to) {
    for (int i = 0; i < models.length; ++i) {
      models[i].appendTo(scale * weights.get(i), vec, to);
    }
    return to;
  }

  @Override
  public RandomVec appendTo(final double scale,
                            final VecDataSet dataSet,
                           final RandomVec dst) {
    for (int i = 0; i < models.length; ++i) {
      models[i].appendTo(scale * weights.get(i), dataSet, dst);
    }
    return dst;
  }

  @Override
  public int dim() {
    return models[ArrayTools.max(models, RandomFunc::dim)].dim();
  }

  public Class<? extends RandomFunc> componentType() {
    return models.length > 0 ? models[0].getClass() : RandomFunc.class;
  }

  @Override
  public boolean equals(final Object o) {
    if (this == o) return true;
    if (!(o instanceof RandomFuncEnsemble)) return false;

    final RandomFuncEnsemble ensemble = (RandomFuncEnsemble) o;

    return Arrays.equals(models, ensemble.models) && weights.equals(ensemble.weights);
  }

  @Override
  public int hashCode() {
    int result = Arrays.hashCode(models);
    result = 31 * result + weights.hashCode();
    return result;
  }
}
