package com.spbsu.exp.dl.functions.unary;

import java.util.Random;

/**
 * jmll
 * ksen
 * 09.December.2014 at 23:53
 */
public class RandomSigmoidFA extends DoubleArrayUnaryFunction {

  private Random random = new Random();

  @Override
  protected void map(final double[] x, final double[] y) {
    for (int i = 0; i < x.length; i++) {
      y[i] = 1. / (1. + Math.exp(-x[i])) > random.nextFloat() ? 1. : 0.;
    }
  }

  @Override
  protected void dMap(final double[] x, final double[] y) {
    throw new UnsupportedOperationException();
  }

}
