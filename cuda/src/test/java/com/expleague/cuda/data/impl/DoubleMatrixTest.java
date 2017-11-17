package com.expleague.cuda.data.impl;

import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertArrayEquals;

/**
 * Created by hrundelb on 20.08.17.
 */
public class DoubleMatrixTest {
  private static final int LENGTH = 10_000;

  private static final float DELTA = 1e-9f;

  private static final Random RANDOM = new Random();

  @Test
  public void testCreate() throws Exception {
    final double[] expected = generateHostDoubleData();

    final DoubleMatrix A = new DoubleMatrix(10, expected);
    final double[] actual = A.get();
    A.destroy();

    assertArrayEquals(expected, actual, DELTA);
  }

  private double[] generateHostDoubleData() {
    final double[] hostData = new double[LENGTH];
    for (int i = 0; i < hostData.length; i++) {
      hostData[i] = RANDOM.nextDouble();
    }
    return hostData;
  }
}
