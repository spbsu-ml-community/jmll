package com.spbsu.exp.dl.functions.unary;

/**
 * jmll
 * ksen
 * 09.December.2014 at 23:50
 */
public class TanhFA extends DoubleArrayUnaryFunction {

  @Override
  protected void map(final double[] x, final double[] y) {
    for (int i = 0; i < x.length; i++) {
      y[i] = Math.tanh(x[i]);
    }
  }

  @Override
  protected void dMap(final double[] x, final double[] y) {
    for (int i = 0; i < x.length; i++) {
      y[i] = Math.pow(1. / Math.cosh(x[i]), 2.);
    }
  }

}
