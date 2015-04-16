package com.spbsu.exp.dl.functions.unary;

/**
 * jmll
 * ksen
 * 14.December.2014 at 00:23
 */
public class BipolarSigmoid extends DoubleArrayUnaryFunction {

  @Override
  protected void map(final double[] x, final double[] y) {
    double value;
    for (int i = 0; i < x.length; i++) {
      value = x[i];
      y[i] = (1. - Math.exp(-value)) / (1.f + Math.exp(-value));
    }
  }

  @Override
  protected void dMap(final double[] x, final double[] y) {
    double value;
    for (int i = 0; i < x.length; i++) {
      value = x[i];
      y[i] = 2 * (Math.exp(value) / Math.pow(Math.exp(value) + 1., 2));
    }
  }

}
