package com.expleague.cuda.data.impl;

import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;

/**
 * Created by hrundelb on 20.08.17.
 */
public class DoubleVectorTest {

  private static final int LENGTH = 10_000;

  private static final float DELTA = 1e-9f;

  private static final Random RANDOM = new Random();

  @Test
  public void testCreate() throws Exception {
    final double[] expected = generateHostDoubleData();

    final DoubleVector a = new DoubleVector(expected);
    final double[] actual = a.get();
    a.destroy();

    assertArrayEquals(expected, actual, DELTA);
  }

  @Test
  public void testSet() throws Exception {
    final double[] data = generateHostDoubleData();

    final DoubleVector a = new DoubleVector(data);
    final double[] expected = generateHostDoubleData();
    a.set(expected);
    final double[] actual = a.get();
    a.destroy();

    assertArrayEquals(expected, actual, DELTA);
  }


  @Test
  public void testReset() {
    final double[] data = generateHostDoubleData();
    final DoubleVector a = new DoubleVector(data);
    System.out.println(Arrays.toString(a.get()));
    a.reproduce();
    System.out.println(Arrays.toString(a.get()));
  }

  private double[] generateHostDoubleData() {
    final double[] hostData = new double[LENGTH];
    for (int i = 0; i < hostData.length; i++) {
      hostData[i] = RANDOM.nextDouble();
    }
    return hostData;
  }
}
