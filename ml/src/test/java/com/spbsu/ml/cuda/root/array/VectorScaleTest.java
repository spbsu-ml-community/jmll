package com.spbsu.ml.cuda.root.array;

import org.junit.Ignore;
import org.junit.Test;

import com.spbsu.ml.cuda.data.impl.FloatVector;
import com.spbsu.ml.cuda.root.array.VectorScale;

import org.junit.Assert;

import java.util.Random;

/**
 * Project jmll
 *
 * @author Ksen
 */
@Ignore
public class VectorScaleTest extends Assert {

  private static final int LENGTH = 10;

  private static final float DELTA = 1e-6f; // (CUDA version) > 6.0 -> EPS = 1e-9

  @Test
  public void testExp() throws Exception {
    final float[] expected = new float[LENGTH];
    final float[] actual = new float[LENGTH];

    final Random random = new Random();
    for (int i = 0; i < LENGTH; i++) {
      final float value = random.nextFloat();

      expected[i] = (float)Math.exp(value);
      actual[i] = value;
    }
    final FloatVector a = new FloatVector(actual);

    VectorScale.fExp(a, a);

    assertArrayEquals(expected, a.get(), DELTA);
    a.destroy();
  }

  @Test
  public void testSigmoid() throws Exception {
    final float[] expected = new float[LENGTH];
    final float[] actual = new float[LENGTH];

    final Random random = new Random();
    for (int i = 0; i < LENGTH; i++) {
      final float value = random.nextFloat();

      expected[i] = 1.f / (1.f + (float)Math.exp(-value));
      actual[i] = value;
    }
    final FloatVector a = new FloatVector(actual);

    VectorScale.fSigmoid(a, a);

    assertArrayEquals(expected, a.get(), DELTA);
    a.destroy();
  }

  @Test
  public void testDerSigmoid() throws Exception {
    final float[] expected = new float[LENGTH];
    final float[] actual = new float[LENGTH];

    final Random random = new Random();
    for (int i = 0; i < LENGTH; i++) {
      final float value = random.nextFloat();

      expected[i] = value - value * value;
      actual[i] = value;
    }
    final FloatVector a = new FloatVector(actual);

    VectorScale.fDerSigmoid(a, a);

    assertArrayEquals(expected, a.get(), DELTA);
    a.destroy();
  }

  @Test
  public void testTanh() throws Exception {
    final float[] expected = new float[LENGTH];
    final float[] actual = new float[LENGTH];

    final Random random = new Random();
    for (int i = 0; i < LENGTH; i++) {
      final float value = random.nextFloat();

      expected[i] = (float)Math.tanh(value);
      actual[i] = value;
    }
    final FloatVector a = new FloatVector(actual);

    VectorScale.fTanh(a, a);

    assertArrayEquals(expected, a.get(), DELTA);
    a.destroy();
  }

  @Test
  public void testNegation() throws Exception {
    final float[] expected = new float[LENGTH];
    final float[] actual = new float[LENGTH];

    final Random random = new Random();
    for (int i = 0; i < LENGTH; i++) {
      final float value = random.nextFloat();

      expected[i] = -value;
      actual[i] = value;
    }
    final FloatVector a = new FloatVector(actual);

    VectorScale.fNegation(a, a);

    assertArrayEquals(expected, a.get(), DELTA);
    a.destroy();
  }

  @Test
  public void testHadamard() throws Exception {
    final float[] expected = new float[LENGTH];
    final float[] actualLeft = new float[LENGTH];
    final float[] actualRight = new float[LENGTH];

    final Random random = new Random();
    for (int i = 0; i < LENGTH; i++) {
      final float valueLeft = random.nextFloat();
      final float valueRight = random.nextFloat();

      expected[i] = valueLeft * valueRight;
      actualLeft[i] = valueLeft;
      actualRight[i] = valueRight;
    }
    final FloatVector a = new FloatVector(actualLeft);
    final FloatVector b = new FloatVector(actualRight);

    VectorScale.fHadamard(a, b, a);

    assertArrayEquals(expected, a.get(), DELTA);
    a.destroy();
    b.destroy();
  }

}
