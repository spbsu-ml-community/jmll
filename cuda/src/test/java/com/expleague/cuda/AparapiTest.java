package com.expleague.cuda;

import com.aparapi.internal.kernel.KernelManager;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.cuda.data.GPUVec;
import org.jetbrains.annotations.TestOnly;
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Created by hrundelb on 20.11.17.
 */
public class AparapiTest {

  @Test
  public void testDevice() {
    System.out.println(KernelManager.instance().bestDevice());
  }

  @Test
  public void testVecSum() {
    final int size = 20;
    Random random = new Random();
    final float[] left = new float[size];
    final float[] right = new float[size];
    for (int i = 0; i < size; i++) {
      left[i] = random.nextFloat();
      right[i] = random.nextFloat();
    }
    System.out.println(Arrays.toString(left));
    System.out.println(Arrays.toString(right));
    float[] sum = AparapiOperations.sum(left, right);
    System.out.println(Arrays.toString(sum));
  }


  @Test
  public void testVecReduce() {
    final int size = 132;
    Random random = new Random();
    final float[] left = new float[size];
    for (int i = 0; i < size; i++) {
      left[i] = 1;
    }
    System.out.println(Arrays.toString(left));
    float[] result = AparapiOperations.vectorReduce(left);
    System.out.println(Arrays.toString(result));
  }

  @Test
  public void testMatrixMult() {
    final int size = 50;
    FastRandom fastRandom = new FastRandom();
    Vec vec1 = new ArrayVec(size * size);
    Vec vec2 = new ArrayVec(size * size);
    for (int i = 0; i < size * size; i++) {
      vec1.set(i, fastRandom.nextGaussian());
      vec2.set(i, fastRandom.nextGaussian());
    }
    Mx mx1 = new VecBasedMx(size, vec1);
    Mx mx2 = new VecBasedMx(size, vec2);
    Mx result = new VecBasedMx(size, size);
    long time = System.nanoTime();
    AparapiOperations.multiplyTo(mx1, mx2, result);
    long diff = System.nanoTime() - time;
    System.out.println("On GPU: " + diff + " ns");
    time = System.nanoTime();
    Mx multiply = MxTools.multiply(mx1, mx2);
    diff = System.nanoTime() - time;
    System.out.println("On CPU: " + diff + " ns");
    assertArrayEquals(result.toArray(), multiply.toArray(), 0.0001);
  }


  @Test
  public void testMatrixMultFloat() {
    final int size = 128;
    FastRandom fastRandom = new FastRandom();
    Vec vec1 = new ArrayVec(size * size);
    Vec vec2 = new ArrayVec(size * size);
    float[] left = new float[size * size];
    float[] right = new float[size * size];
    for (int i = 0; i < size * size; i++) {
      float v1 = fastRandom.nextFloat();
      vec1.set(i, v1);
      left[i] = v1;
      float v2 = fastRandom.nextFloat();
      vec2.set(i, v2);
      right[i] = v2;
    }
    Mx mx1 = new VecBasedMx(size, vec1);
    Mx mx2 = new VecBasedMx(size, vec2);
    float[] result = new float[size * size];
    System.out.println("Start GPU");
    long time = System.nanoTime();
    AparapiOperations.multiplyTo(left, right, result, size);
    long diff1 = System.nanoTime() - time;
    System.out.println("On GPU: " + diff1 + " ns");
    System.out.println("Start CPU");
    time = System.nanoTime();
    Mx multiply = MxTools.multiply(mx1, mx2);
    long diff2 = System.nanoTime() - time;
    System.out.println("On CPU: " + diff2 + " ns");
    double v = (double) diff2 / diff1;
    System.out.printf("%.1fx faster", v);
    assertArrayEquals(GPUVec.convert(result), multiply.toArray(), 0.01);
  }


  @Test
  public void testMatrixTransposeFloat() {
    final int size = 128;
    FastRandom fastRandom = new FastRandom();
    Vec vec1 = new ArrayVec(size * size);
    float[] left = new float[size * size];
    for (int i = 0; i < size * size; i++) {
      float v1 = fastRandom.nextFloat();
      vec1.set(i, v1);
      left[i] = v1;
    }
    Mx mx1 = new VecBasedMx(size, vec1);
    float[] result = new float[size * size];
    System.out.println("Start GPU");
    long time = System.nanoTime();
    AparapiOperations.transpose(left, result, size);
    long diff1 = System.nanoTime() - time;
    System.out.println("On GPU: " + diff1 + " ns");
    System.out.println("Start CPU");
    time = System.nanoTime();
    Mx mxResult = MxTools.transpose(mx1);
    long diff2 = System.nanoTime() - time;
    System.out.println("On CPU: " + diff2 + " ns");
    double v = (double) diff2 / diff1;
    System.out.printf("%.1fx faster", v);
    //System.out.println(new VecBasedMx(size, new ArrayVec(convert(result))));
    //System.out.println(mxResult);
    assertArrayEquals(GPUVec.convert(result), mxResult.toArray(), 0.01);
  }
}
