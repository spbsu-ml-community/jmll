package com.spbsu.exp.dl.functions.binary;

/**
 * jmll
 * ksen
 * 13.December.2014 at 12:39
 */
public class HadamardFA extends DoubleArrayBinaryFunction {

  @Override
  protected void map(final double[] x, final double[] z, final double[] y) {
    for (int i = 0; i < x.length; i++) {
      y[i] = x[i] * z[i];
    }
  }

  @Override
  protected void dMap(double[] x, double[] z, double[] y) {
    throw new UnsupportedOperationException();
  }

}
