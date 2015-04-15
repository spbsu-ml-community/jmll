package com.spbsu.ml.cuda;

import org.junit.Ignore;
import org.junit.Test;

import jcuda.driver.CUdeviceptr;

import org.junit.Assert;

import java.util.Random;

/**
 * jmll
 * ksen
 * 08.April.2015 at 00:32
 */
@Ignore
public class JCudaMemoryTest extends Assert {

  private static final int LENGTH = 10_000;
  private static final double EPS = 1e-9;

  private static final Random RANDOM = new Random();

  @Test
  public void testDoubleAlloCopyDestr() throws Exception {
    final double[] expected = generateHostData();

    final CUdeviceptr devicePointer = JCudaMemory.alloCopy(expected);
    final double[] actual = JCudaMemory.copyDestr(devicePointer, expected.length);

    assertArrayEquals(expected, actual, EPS);
  }

  @Test
  public void testDoubleAlloCopyDestrParam() throws Exception {
    final double[] expected = generateHostData();

    final CUdeviceptr devicePointer = JCudaMemory.alloCopy(expected);
    final double[] actual = new double[expected.length];
    JCudaMemory.copyDestr(devicePointer, actual);

    assertArrayEquals(expected, actual, EPS);
  }

  private double[] generateHostData() {
    final double[] hostData = new double[LENGTH];
    for (int i = 0; i < hostData.length; i++) {
      hostData[i] = RANDOM.nextDouble();
    }
    return hostData;
  }

}
