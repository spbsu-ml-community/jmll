package com.spbsu.exp.dl.functions.unary;

/**
 * jmll
 * ksen
 * 09.December.2014 at 23:34
 */
public class SigmoidFA extends DoubleArrayUnaryFunction {

  @Override
  protected void map(final double[] x, final double[] y) {
    for (int i = 0; i < x.length; i++) {
      y[i] = 1. / (1. + Math.exp(-x[i]));
    }
  }

  @Override
  protected void dMap(final double[] x, final double[] y) {
    for (int i = 0; i < x.length; i++) {
      y[i] = x[i] - x[i] * x[i];
    }
  }

}
