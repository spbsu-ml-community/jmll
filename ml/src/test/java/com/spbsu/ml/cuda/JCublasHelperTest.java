package com.spbsu.ml.cuda;

import org.junit.AfterClass;
import org.junit.Ignore;
import org.junit.Test;

import org.junit.Assert;

import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.ml.cuda.data.impl.FloatMatrix;

/**
 * Project jmll
 *
 * @author Ksen
 */
@Ignore
public class JCublasHelperTest extends Assert {

  private final static double DELTA = 1e-9;
  private final static float EPS = 1e-9f;

  @AfterClass
  public static void shutdown() {
    JCublasHelper.shutdown();
  }

  @Test
  public void testMax() throws Exception {
    final ArrayVec a = new ArrayVec(1, 2, 3, 4, 5);

    final int maxA = JCublasHelper.max(a);
    assertEquals(4, maxA);

    final ArrayVec b = new ArrayVec(10, 5, 1);

    final int maxB = JCublasHelper.max(b);
    assertEquals(0, maxB);

    final ArrayVec c = new ArrayVec(-10030, -91283476, 0, 123, -10);
    final int maxC = JCublasHelper.max(c);
    assertEquals(1, maxC);
  }

  @Test
  public void testMin() throws Exception {
    final ArrayVec a = new ArrayVec(1, 2, 3, 4, 5);

    final int minA = JCublasHelper.min(a);
    assertEquals(0, minA);

    final ArrayVec b = new ArrayVec(10, 5, 1);

    final int minB = JCublasHelper.min(b);
    assertEquals(2, minB);

    final ArrayVec c = new ArrayVec(-10030, -91283476, 0, 123, -10);
    final int minC = JCublasHelper.min(c);
    assertEquals(2, minC);
  }

  @Test
  public void testDot() throws Exception {
    final ArrayVec a = new ArrayVec(1, 2, 3);
    final ArrayVec b = new ArrayVec(1, 2, 3);

    final double dot1 = JCublasHelper.dot(a, b);
    assertEquals(1 * 1 + 2 * 2 + 3 * 3, dot1, DELTA);

    final ArrayVec c = new ArrayVec(0.124, 87123.18723, 0.00000000000000000001);
    final ArrayVec d = new ArrayVec(10000000001., 70, 1);

    final double dot2 = JCublasHelper.dot(c, d);
    assertEquals(0.124 * 10000000001. + 87123.18723 * 70 + 0.00000000000000000001 * 1, dot2, DELTA);

    final ArrayVec e = new ArrayVec(-1082364, 0.10000000000000001, -0.25);
    final ArrayVec f = new ArrayVec(0.1, -10000000000., -99817243);

    final double dot3 = JCublasHelper.dot(e, f);
    assertEquals(-1082364 * 0.1 + 0.10000000000000001 * -10000000000. + -0.25 * -99817243, dot3, DELTA);
  }

  @Test
  public void testManhattan() throws Exception {
    final ArrayVec a = new ArrayVec(1, 2, 3);

    final double manhattan1 = JCublasHelper.manhattan(a);
    assertEquals(1 + 2 + 3, manhattan1, DELTA);

    final ArrayVec b = new ArrayVec(1., 1.01, 1.00000001, 1.000000000000001);

    final double manhattan2 = JCublasHelper.manhattan(b);
    assertEquals(1. + 1.01 + 1.00000001 + 1.000000000000001, manhattan2, DELTA);

    final ArrayVec c = new ArrayVec(-1082364, 0.10000000000000001, -0.25);

    final double manhattan3 = JCublasHelper.manhattan(c);
    assertEquals(1082364 + 0.10000000000000001 + 0.25, manhattan3, DELTA);
  }

//  @Test //todo(ksenon): failed
//  public void testEuclidean() throws Exception {
//    final ArrayVec a = new ArrayVec(1, 2, 3);
//
//    final double euclidean1 = JCublasHelper.euclidean(a);
//    assertEquals(pow(1) + pow(2) + pow(3), euclidean1, DELTA);
//
//    final ArrayVec b = new ArrayVec(1., 1.01, 1.00000001, 1.000000000000001);
//
//    final double euclidean2 = JCublasHelper.euclidean(b);
//    assertEquals(pow(1.) + pow(1.01) + pow(1.00000001) + pow(1.000000000000001), euclidean2, DELTA);
//
//    final ArrayVec c = new ArrayVec(-1082364, 0.10000000000000001, -0.25);
//
//    final double euclidean3 = JCublasHelper.euclidean(c);
//    assertEquals(pow(-1082364) + pow(0.10000000000000001) + pow(-0.25), euclidean3, DELTA);
//  }
//
//  private double pow(final double x) {
//    return Math.pow(x, 2);
//  }

  @Test
  public void testScale() throws Exception {
    final ArrayVec a = new ArrayVec(1, 2, 3);

    final ArrayVec b = JCublasHelper.scale(10, a);
    VecTools.scale(a, 10);

    assertTrue(a.equals(b));

    final ArrayVec c = new ArrayVec(1000000000000., 0, 0);

    final ArrayVec d = JCublasHelper.scale(-1, c);
    VecTools.scale(c, -1);

    assertTrue(c.equals(d));

    final ArrayVec e = new ArrayVec(-10, 20, 0, 0, 0.000000000000000001);

    final ArrayVec f = JCublasHelper.scale(-0.01111111222, e);
    VecTools.scale(e, -0.01111111222);

    assertTrue(e.equals(f));
  }

  @Test
  public void testSum() throws Exception {
    final ArrayVec a = new ArrayVec(1, 2, 3);
    final ArrayVec b = new ArrayVec(1, 2, 3);

    final ArrayVec c = JCublasHelper.sum(a, b);
    final ArrayVec d = VecTools.sum(a, b);

    assertTrue(c.equals(d));

    final ArrayVec a2 = new ArrayVec(-1000.1010101);
    final ArrayVec b2 = new ArrayVec(0.);

    final ArrayVec c2 = JCublasHelper.sum(a2, b2);
    final ArrayVec d2 = VecTools.sum(a2, b2);

    assertTrue(c2.equals(d2));

    final ArrayVec a3 = new ArrayVec(-10, -20, 0, 0, 1, 0, 10, 0);
    final ArrayVec b3 = new ArrayVec(0, 0, 1, 1, -1, -1, 0, 0);

    final ArrayVec c3 = JCublasHelper.sum(a3, b3);
    final ArrayVec d3 = VecTools.sum(a3, b3);

    assertTrue(c3.equals(d3));
  }

  @Test
  public void testDevFloatMultMxMx() throws Exception {
    final float[] leftData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    final float[] rightData = new float[15];

    final FloatMatrix A = new FloatMatrix(3, leftData);
    final FloatMatrix B = new FloatMatrix(5, rightData);
    final FloatMatrix C = new FloatMatrix(3, new float[9]);

    JCublasHelper.mult(A, false, B, false, C);

    final float[] expected = C.get();

    assertArrayEquals(new float[9], expected, EPS);

    A.destroy();
    B.destroy();
    C.destroy();
  }

  @Test
  public void testDevFloatMultMxTMx() throws Exception {
    final float[] leftData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    final float[] rightData = new float[15];

    final FloatMatrix A = new FloatMatrix(3, leftData);
    final FloatMatrix B = new FloatMatrix(3, rightData);
    final FloatMatrix C = new FloatMatrix(3, new float[9]);

    JCublasHelper.mult(A, false, B, true, C);

    final float[] expected = C.get();

    assertArrayEquals(new float[9], expected, EPS);

    A.destroy();
    B.destroy();
    C.destroy();
  }

  @Test
  public void testDevFloatMultMxMxT() throws Exception {
    final float[] leftData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    final float[] rightData = new float[15];

    final FloatMatrix A = new FloatMatrix(5, leftData);
    final FloatMatrix B = new FloatMatrix(5, rightData);
    final FloatMatrix C = new FloatMatrix(3, new float[9]);

    JCublasHelper.mult(A, true, B, false, C);

    final float[] expected = C.get();

    assertArrayEquals(new float[9], expected, EPS);

    A.destroy();
    B.destroy();
    C.destroy();
  }

  @Test
  public void testDevFloatMultMxTMxT() throws Exception {
    final float[] leftData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    final float[] rightData = new float[21];

    final FloatMatrix A = new FloatMatrix(3, leftData);
    final FloatMatrix B = new FloatMatrix(7, rightData);
    final FloatMatrix C = new FloatMatrix(5, new float[35]);

    JCublasHelper.mult(A, true, B, true, C);

    final float[] expected = C.get();

    assertArrayEquals(new float[35], expected, EPS);

    A.destroy();
    B.destroy();
    C.destroy();
  }

}
