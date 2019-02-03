package com.expleague.cuda.data;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.cuda.JCudaHelper;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertArrayEquals;

/**
 * Created by hrundelb on 29.08.17.
 */
public class GPUVecTest {

  private static final int LENGTH = 10_000;

  private static final float DELTA = 1e-9f;

  private static final Random RANDOM = new Random();

  @BeforeClass
  public static void initCuda() {
    Assume.assumeNoException(JCudaHelper.checkInstance());
  }

  @Test
  public void testCreate() {
    final float[] expected = generateHostFloatData();

    final GPUVec a = new GPUVec(expected);
    final float[] actual = a.toFloatArray();

    assertArrayEquals(expected, actual, DELTA);
  }

  @Test
  public void testSub() {
    final float[] data = generateHostFloatData();
    final double[] doubles = new double[data.length];
    for (int i = 0; i < doubles.length; i++) {
      doubles[i] = data[i];
    }

    final Vec a = new ArrayVec(doubles);
    final Vec b = new GPUVec(data);

    int start = RANDOM.nextInt(LENGTH / 2);
    int len = RANDOM.nextInt(LENGTH / 2);
    Vec aSub = a.sub(start, len);
    Vec bSub = b.sub(start, len);

    assertArrayEquals(aSub.toArray(), bSub.toArray(), DELTA);
  }


  private float[] generateHostFloatData() {
    final float[] hostData = new float[LENGTH];
    for (int i = 0; i < hostData.length; i++) {
      hostData[i] = RANDOM.nextFloat();
    }
    return hostData;
  }
}
