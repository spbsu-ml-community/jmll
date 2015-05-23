package com.spbsu.ml.cuda;

import org.junit.Ignore;
import org.junit.Test;

import com.spbsu.ml.cuda.data.impl.FloatVector;

import jcuda.driver.CUdeviceptr;
import org.junit.Assert;

import java.util.Random;

/**
 * Project jmll
 *
 * @author Ksen
 */
@Ignore
public class JCudaMemoryTest extends Assert {

  private static final int LENGTH = 10_000;

  private static final float DELTA = 1e-9f;
  private static final double EPS = 1e-12;

  private static final Random RANDOM = new Random();

  @Test
  public void testFloatAlloCopyDestr() throws Exception {
    final float[] actual = generateHostFloatData();

    final CUdeviceptr devicePointer = JCudaMemory.alloCopy(actual);
    final float[] expected = JCudaMemory.copyFloatsDestr(devicePointer, actual.length);

    assertArrayEquals(actual, expected, DELTA);
  }

  @Test
  public void testInsertFloats() throws Exception {
    final float[] sourceData = {1, 2, 3, 4, 5};
    final FloatVector source = new FloatVector(sourceData);

    final float[] destinationData = new float[5];
    final FloatVector destination = new FloatVector(destinationData);

    JCudaMemory.insertFloats(source.devicePointer, destination.devicePointer, 5);

    assertArrayEquals(sourceData, destination.get(), DELTA);
  }

  @Test
  public void testInsertShiftedFloats() throws Exception {
    final float[] sourceData = {1, 2, 3, 4, 5};
    final FloatVector source = new FloatVector(sourceData);

    final float[] destinationData = new float[2];
    final FloatVector destination = new FloatVector(destinationData);

    JCudaMemory.insertFloats(source.devicePointer, 3, destination.devicePointer, 0, 2);

    assertArrayEquals(new float[]{4, 5}, destination.get(), DELTA);
  }

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

  private float[] generateHostFloatData() {
    final float[] hostData = new float[LENGTH];
    for (int i = 0; i < hostData.length; i++) {
      hostData[i] = RANDOM.nextFloat();
    }
    return hostData;
  }

  private double[] generateHostData() {
    final double[] hostData = new double[LENGTH];
    for (int i = 0; i < hostData.length; i++) {
      hostData[i] = RANDOM.nextDouble();
    }
    return hostData;
  }

}
