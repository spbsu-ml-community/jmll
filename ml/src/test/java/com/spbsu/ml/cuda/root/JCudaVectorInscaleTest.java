package com.spbsu.ml.cuda.root;

import org.junit.Ignore;
import org.junit.Test;

import org.junit.Assert;

import java.util.Random;

/**
 * jmll
 * ksen
 * 03.April.2015 at 08:12
 */
@Ignore
public class JCudaVectorInscaleTest extends Assert {

  private static final int LENGTH = 10;

  private static final double EPS = 1e-6; // (CUDA version) > 6.0 -> EPS = 1e-9

  @Test
  public void testExp() throws Exception {
    final double[] expected = new double[LENGTH];
    final double[] actual = new double[LENGTH];

    final Random random = new Random();
    for (int i = 0; i < LENGTH; i++) {
      final double value = random.nextDouble();

      expected[i] = Math.exp(value);
      actual[i] = value;
    }

    JCudaVectorInscale.exp(actual);

    assertArrayEquals(expected, actual, EPS);
  }

  @Test
  public void testSigmoid() throws Exception {
    final double[] expected = new double[LENGTH];
    final double[] actual = new double[LENGTH];

    final Random random = new Random();
    for (int i = 0; i < LENGTH; i++) {
      final double value = random.nextDouble();

      expected[i] = 1. / (1. + Math.exp(-value));
      actual[i] = value;
    }

    JCudaVectorInscale.sigmoid(actual);

    assertArrayEquals(expected, actual, EPS);
  }

  @Test
  public void testTanh() throws Exception {
    final double[] expected = new double[LENGTH];
    final double[] actual = new double[LENGTH];

    final Random random = new Random();
    for (int i = 0; i < LENGTH; i++) {
      final double value = random.nextDouble();

      expected[i] = Math.tanh(value);
      actual[i] = value;
    }

    JCudaVectorInscale.tanh(actual);

    assertArrayEquals(expected, actual, EPS);
  }

}
