package com.spbsu.ml.cli.gridsearch;

import java.util.Arrays;

/**
* User: qdeee
* Date: 29.03.15
*/
public class OptimumHolder {
  private final Object[] fixedParameters;
  private final double[] metricValues;
  private final double targetValue;

  public OptimumHolder(final Object[] fixedParameters, final double[] metricValues, final double targetValue) {
    this.fixedParameters = fixedParameters;
    this.metricValues = metricValues;
    this.targetValue = targetValue;
  }

  public boolean isBetterThan(final OptimumHolder holder, final int metricIndex) {
    return holder == null || this.metricValues[metricIndex] > holder.metricValues[metricIndex];
  }

  @Override
  public String toString() {
    return "OptimumHolder{" +
        "fixedParameters=" + Arrays.toString(fixedParameters) +
        ", metricValues=" + Arrays.toString(metricValues) +
        ", targetValue=" + targetValue +
        '}';
  }
}
