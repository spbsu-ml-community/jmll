package com.spbsu.ml.cuda;

import com.spbsu.commons.io.StreamTools;
import org.junit.Test;

import org.junit.Assert;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;

import java.io.*;
import java.util.Arrays;

/**
 * jmll
 * ksen
 * 28.February.2015 at 22:59
 */
public class JCublasHelperTest extends Assert {

  private final static double DELTA = 1e-9;

  @Test
  public void testMax() throws Exception {
    final Vec a = new ArrayVec(1, 2, 3, 4, 5);

    final int maxA = JCublasHelper.max(a);
    assertEquals(4, maxA);

    final Vec b = new ArrayVec(10, 5, 1);

    final int maxB = JCublasHelper.max(b);
    assertEquals(0, maxB);

    final Vec c = new ArrayVec(-10030, -91283476, 0, 123, -10);
    final int maxC = JCublasHelper.max(c);
    assertEquals(1, maxC);
  }

  @Test
  public void testMin() throws Exception {
    final Vec a = new ArrayVec(1, 2, 3, 4, 5);

    final int minA = JCublasHelper.min(a);
    assertEquals(0, minA);

    final Vec b = new ArrayVec(10, 5, 1);

    final int minB = JCublasHelper.min(b);
    assertEquals(2, minB);

    final Vec c = new ArrayVec(-10030, -91283476, 0, 123, -10);
    final int minC = JCublasHelper.min(c);
    assertEquals(2, minC);
  }

  @Test
  public void testDot() throws Exception {
    final Vec a = new ArrayVec(1, 2, 3);
    final Vec b = new ArrayVec(1, 2, 3);

    final double dot1 = JCublasHelper.dot(a, b);
    assertEquals(1 * 1 + 2 * 2 + 3 * 3, dot1, DELTA);

    final Vec c = new ArrayVec(0.124, 87123.18723, 0.00000000000000000001);
    final Vec d = new ArrayVec(10000000001., 70, 1);

    final double dot2 = JCublasHelper.dot(c, d);
    assertEquals(0.124 * 10000000001. + 87123.18723 * 70 + 0.00000000000000000001 * 1, dot2, DELTA);

    final Vec e = new ArrayVec(-1082364, 0.10000000000000001, -0.25);
    final Vec f = new ArrayVec(0.1, -10000000000., -99817243);

    final double dot3 = JCublasHelper.dot(e, f);
    assertEquals(-1082364 * 0.1 + 0.10000000000000001 * -10000000000. + -0.25 * -99817243, dot3, DELTA);
  }

  @Test
  public void testManhattan() throws Exception {
    final Vec a = new ArrayVec(1, 2, 3);

    final double manhattan1 = JCublasHelper.manhattan(a);
    assertEquals(1 + 2 + 3, manhattan1, DELTA);

    final Vec b = new ArrayVec(1., 1.01, 1.00000001, 1.000000000000001);

    final double manhattan2 = JCublasHelper.manhattan(b);
    assertEquals(1. + 1.01 + 1.00000001 + 1.000000000000001, manhattan2, DELTA);

    final Vec c = new ArrayVec(-1082364, 0.10000000000000001, -0.25);

    final double manhattan3 = JCublasHelper.manhattan(c);
    assertEquals(1082364 + 0.10000000000000001 + 0.25, manhattan3, DELTA);
  }

//  @Test //todo(ksenon): failed
//  public void testEuclidean() throws Exception {
//    final Vec a = new ArrayVec(1, 2, 3);
//
//    final double euclidean1 = JCublasHelper.euclidean(a);
//    assertEquals(pow(1) + pow(2) + pow(3), euclidean1, DELTA);
//
//    final Vec b = new ArrayVec(1., 1.01, 1.00000001, 1.000000000000001);
//
//    final double euclidean2 = JCublasHelper.euclidean(b);
//    assertEquals(pow(1.) + pow(1.01) + pow(1.00000001) + pow(1.000000000000001), euclidean2, DELTA);
//
//    final Vec c = new ArrayVec(-1082364, 0.10000000000000001, -0.25);
//
//    final double euclidean3 = JCublasHelper.euclidean(c);
//    assertEquals(pow(-1082364) + pow(0.10000000000000001) + pow(-0.25), euclidean3, DELTA);
//  }
//
//  private double pow(final double x) {
//    return Math.pow(x, 2);
//  }


//  @Test
//  public void testName() throws Exception {
//    final int[] array = new int[300000000];
//    for (int r = 0; r < 1000000; r++) {
//      long begin = System.nanoTime();
//
//      int max = Integer.MIN_VALUE;
//      for (int i = 0; i < array.length; i++) {
//        if (max < array[i]) {
//          max = array[i];
//        }
//      }
//
//      for (int exp = 1; max / exp > 0; exp *= 10) {
//        int[] output = new int[array.length];
//        int[] counters = new int[10];
//        for (int i = 0; i < array.length; i++) {
//          counters[(array[i] / exp) % 10]++;
//        }
//        for (int i = 1; i < 10; i++) {
//          counters[i] += counters[i - 1];
//        }
//        for (int i = array.length - 1; i > -1; i--) {
//          output[--counters[(array[i] / exp) % 10]] = array[i];
//        }
//        System.arraycopy(output, 0, array, 0, array.length);
//      }
//
//      System.out.println(System.nanoTime() - begin);
//    }
//  }
//
//  @Test
//  public void testName2() throws Exception {
//    final int[] array = new int[100000000];
//    for (int r = 0; r < 1000000; r++) {
//      long begin = System.nanoTime();
//      Arrays.sort(array);
//      System.out.println(System.nanoTime() - begin);
//    }
//  }

  @Test
  public void testName() throws Exception {
    System.out.println(File.separator);
  }

}
