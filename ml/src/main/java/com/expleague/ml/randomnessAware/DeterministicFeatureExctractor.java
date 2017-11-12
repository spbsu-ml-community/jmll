package com.expleague.ml.randomnessAware;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.distributions.RandomVecBuilder;
import com.expleague.ml.distributions.parametric.DeltaFunction;
import com.expleague.ml.distributions.parametric.impl.DeltaDistributionVec;

/**
 * Created by noxoomo on 05/11/2017.
 */
public class DeterministicFeatureExctractor implements VecRandomFeatureExtractor<DeltaFunction> {
  private final int dim;
  private final int colId;

  public DeterministicFeatureExctractor(final int dim,
                                        final int colId) {
    this.dim = dim;
    this.colId = colId;
  }

  @Override
  public DeltaFunction compute(final Vec featuresVec) {
    return () -> featuresVec.get(colId);
  }

  @Override
  public RandomVec<DeltaFunction> apply(final VecDataSet dataSet) {
    return new DeltaDistributionVec(dataSet.data().col(colId));
  }

  @Override
  public int dim() {
    return dim;
  }

  @Override
  public RandomVecBuilder<DeltaFunction> randomVecBuilder() {
    return new DeltaFunction.VecBuilder();
  }

  @Override
  public String toString() {
    return "f" +colId;
  }
}
