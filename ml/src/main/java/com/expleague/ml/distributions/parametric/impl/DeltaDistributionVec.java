package com.expleague.ml.distributions.parametric.impl;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.distributions.RandomVecBuilder;
import com.expleague.ml.distributions.parametric.DeltaFunction;
import com.expleague.ml.distributions.samplers.RandomVecSampler;

/**
 * Created by noxoomo on 01/11/2017.
 */
public class DeltaDistributionVec implements RandomVec<DeltaFunction> {
  private final Vec data;

  public DeltaDistributionVec(final Vec data) {
    this.data = data;
  }

  @Override
  public DeltaFunction randomVariable(final int idx) {
    return new CoordinateImpl(idx);
  }

  @Override
  public RandomVecBuilder<DeltaFunction> builder() {
    return new DeltaFunction.VecBuilder();
  }

  @Override
  public RandomVec<DeltaFunction> setRandomVariable(final int idx, final DeltaFunction var) {
     data.set(idx, var.value());
     return this;
  }

  private RandomVecSampler sampler = new RandomVecSampler() {
    @Override
    public final double instance(final FastRandom random, final int i) {
      return data.get(i);
    }

    @Override
    public final int dim() {
      return data.dim();
    }
  };

  @Override
  public RandomVecSampler sampler() {
    return sampler;
  }

  @Override
  public Vec expectationTo(final Vec to) {
    VecTools.copyTo(data, to, 0);
    return to;
  }

  @Override
  public int dim() {
    return data.dim();
  }

  @Override
  public double expectation(final int idx) {
    return data.get(idx);
  }

  @Override
  public double cumulativeProbability(final int idx, final double x) {
    return data.get(idx) < x ? 0 : 1;
  }

  class CoordinateImpl implements DeltaFunction {
    final int idx;

    CoordinateImpl(final int idx) {
      this.idx = idx;
    }

    @Override
    public double value() {
      return data.get(idx);
    }

  }
}
