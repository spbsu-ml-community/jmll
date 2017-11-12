package com.expleague.ml.func;

import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.util.ArrayTools;

import java.util.Arrays;
import java.util.List;

/**
 * User: solar
 * Date: 26.11.12
 * Time: 15:56
 */
public class Ensemble<F extends Trans> extends Trans.Stub {
  public final F[] models;
  public final Vec weights;

  public Ensemble(final F[] models,
                  final Vec weights) {
    this.models = models;
    this.weights = weights;
  }

  public Ensemble(final List<F> weakModels,
                  final double step) {
    this(ArrayTools.toArray(weakModels), VecTools.fill(new ArrayVec(weakModels.size()), step));
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
    return models[ArrayTools.max(models, Trans::ydim)].ydim();
  }

  @Override
  public Trans gradient() {
    return new Ensemble<>(ArrayTools.map(models, Trans.class, Trans::gradient), weights);
  }

  public Class<? extends Trans> componentType() {
    return models.length > 0 ? models[0].getClass() : Trans.class;
  }

  @Override
  public Vec trans(final Vec x) {
    final Vec result = new ArrayVec(ydim());
    for (int i = 0; i < models.length; i++) {
      VecTools.append(result, VecTools.scale(models[i].trans(x), weights.get(i)));
    }
    return result;
  }

  @Override
  public boolean equals(final Object o) {
    if (this == o) return true;
    if (!(o instanceof Ensemble)) return false;

    final Ensemble ensemble = (Ensemble) o;

    return Arrays.equals(models, ensemble.models) && weights.equals(ensemble.weights);
  }

  @Override
  public int hashCode() {
    int result = Arrays.hashCode(models);
    result = 31 * result + weights.hashCode();
    return result;
  }
}
