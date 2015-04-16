package com.spbsu.exp.dl.functions.unary;

/**
 * jmll
 * ksen
 * 09.December.2014 at 23:32
 */
public class HeavisideFA extends DoubleArrayUnaryFunction {

  @Override
  protected void map(final double[] x, final double[] y) {
    for (int i = 0; i < x.length; i++) {
      y[i] = x[i] < 0. ? 0. : 1.;
    }
  }

  @Override
  protected void dMap(final double[] x, final double[] y) {
    throw new UnsupportedOperationException();
  }

}
