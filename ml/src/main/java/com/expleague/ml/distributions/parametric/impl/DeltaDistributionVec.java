package com.expleague.ml.distributions.parametric.impl;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.distributions.RandomVecBuilder;
import com.expleague.ml.distributions.parametric.DeltaFunction;

/**
 * Created by noxoomo on 01/11/2017.
 */
public class DeltaDistributionVec extends RandomVec.CoordinateIndependentStub implements RandomVec {
  private final Vec data;

  public DeltaDistributionVec(final Vec data) {
    this.data = data;
  }

  @Override
  public RandomVariable at(int idx) {
    return new CoordinateImpl(this, idx);
  }

  @Override
  public double instance(int idx, FastRandom random) {
    return data.get(idx);
  }

  @Override
  public double logDensity(int idx, double value) {
    return data.get(idx) == value ? 0 : Double.NEGATIVE_INFINITY;
  }


  @Override
  public int length() {
    return data.dim();
  }

  public double expectation(final int idx) {
    return data.get(idx);
  }

  @Override
  public double cdf(final int idx, final double x) {
    return x < data.get(idx) ? 0 : 1;
  }


  class CoordinateImpl  extends RandomVec.CoordinateProjectionStub<DeltaDistributionVec> implements DeltaFunction {

    CoordinateImpl(final DeltaDistributionVec owner, final int idx) {
      super(owner, idx);
    }

    @Override
    public double value() {
      return data.get(idx);
    }
  }

  static public class VecBuilder implements RandomVecBuilder<DeltaFunction> {
    final com.expleague.commons.math.vectors.impl.vectors.VecBuilder vecBuilder = new com.expleague.commons.math.vectors.impl.vectors.VecBuilder();

    @Override
    public RandomVecBuilder<DeltaFunction> add(final DeltaFunction distribution) {
      vecBuilder.append(distribution.value());
      return this;
    }

    @Override
    public RandomVec build() {
      return new DeltaDistributionVec(vecBuilder.build());
    }
  }


}
