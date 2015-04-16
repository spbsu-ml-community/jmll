package com.spbsu.exp.dl.functions.unary;

/**
 * jmll
 * ksen
 * 10.December.2014 at 00:02
 */
public class TanhOptimalFA extends DoubleArrayUnaryFunction {

  @Override
  protected void map(final double[] x, final double[] y) {
    final double alpha = 2. / 3.;
    for (int i = 0; i < x.length; i++) {
      y[i] = 1.7159f * (double)Math.tanh(alpha * x[i]);
    }
  }

  @Override
  protected void dMap(final double[] x, final double[] y) {
    final double alpha = 1.7159f * 2. / 3.;
    final double beta = 1. / Math.pow(1.7159, 2);
    for (int i = 0; i < x.length; i++) {
      y[i] = alpha * (1. - beta * Math.pow(x[i], 2));
    }
  }

}
