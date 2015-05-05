package com.spbsu.exp.dl.functions.unary;

/**
 * jmll
 * ksen
 * 10.December.2014 at 00:58
 */
public class IdenticalFA extends DoubleArrayUnaryFunction {

  @Override
  protected void map(final double[] x, final double[] y) {
    System.arraycopy(x, 0, y, 0, x.length);
  }

  @Override
  protected void dMap(final double[] x, double[] y) {
    for (int i = 0; i < x.length; i++) {
      y[i] = 1.;
    }
  }

}
