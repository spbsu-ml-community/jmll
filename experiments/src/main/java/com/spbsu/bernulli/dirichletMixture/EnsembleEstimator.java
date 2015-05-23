package com.spbsu.bernulli.dirichletMixture;

import com.spbsu.commons.util.ArrayTools;

import java.util.function.IntToDoubleFunction;

public class EnsembleEstimator {
  private final double[] meanSums;
  private int count; //normalization for estimator

  public EnsembleEstimator(int len) {
    this.meanSums = new double[len];
  }

  public final double[] get() {
    double[] result = new double[meanSums.length];
    System.arraycopy(meanSums, 0, result, 0, result.length);
    for (int i = 0; i < meanSums.length; ++i)
      result[i] /= count;
    return result;
  }

  public final void clear() {
    ArrayTools.fill(meanSums, 0);
    count = 0;
  }

  public final void add(IntToDoubleFunction estimator) {
    count ++;
    for (int i=0; i < meanSums.length;++i)
      meanSums[i] += estimator.applyAsDouble(i);
  }
}

