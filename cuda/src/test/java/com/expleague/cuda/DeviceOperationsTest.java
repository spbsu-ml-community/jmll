package com.expleague.cuda;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.cuda.data.GPUMx;
import com.expleague.cuda.data.GPUVec;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Created by hrundelb on 30.10.17.
 */
public class DeviceOperationsTest {

  private final static double DELTA = 1e-3;
  private final static float EPS = 1e-3f;
  private final static Random random = new Random();

  private GPUVec[] vectors;
  private Vec[] vecs;
  public static final int SIZE = 10000;
  public static final int COUNT = 10;

  @Before
  public void init() {
    vectors = new GPUVec[COUNT];
    vecs = new Vec[COUNT];

    for (int j = 0; j < COUNT; j++) {
      double[] doubles = new double[SIZE];
      for (int i = 0; i < SIZE; i++) {
        doubles[i] = random.nextDouble();
      }
      vectors[j] = new GPUVec(doubles);
      vecs[j] = new ArrayVec(doubles);
    }
  }

  @AfterClass
  public static void shutdown() {
    JCublasHelper.shutdown();
  }

  @Test
  public void testDevFloatMultMM() {
    int size = 1000000;
    int rows = 1000;
    int iterations = 10;

    Random random = new Random();
    final GPUMx[] doubleMatrices = new GPUMx[iterations];
    final Mx[] mxes = new Mx[iterations];
    for (int i = 0; i < iterations; i++) {
      final double[] doubles = new double[size];
      for (int j = 0; j < doubles.length; j++) {
        doubles[j] = random.nextDouble() * 0.01;
      }
      doubleMatrices[i] = new GPUMx(rows, new GPUVec(doubles));
      mxes[i] = new VecBasedMx(rows, new ArrayVec(doubles));
    }

    System.out.println("Start on GPU");
    long timeGPU = System.currentTimeMillis();
    for (int i = 0; i < iterations - 2; i++) {
      DeviceOperations.multiply(doubleMatrices[i], doubleMatrices[i + 2], doubleMatrices[i + 1]);
    }
    System.out.println(String.format("Stop on GPU. Time: %s ms",
        System.currentTimeMillis() - timeGPU));
    GPUMx A = doubleMatrices[iterations - 2];
    System.out.println("Result:\n" + arraytoString(A.toArray(), 20));



    System.out.println("Start on CPU");
    long time = System.currentTimeMillis();
    for (int i = 0; i < iterations - 2; i++) {
      mxes[i + 1] = MxTools.multiply(mxes[i], mxes[i + 2]);
    }
    System.out.println(String.format("Stop on CPU. Time: %s ms",
        System.currentTimeMillis() - time));

    Mx a = mxes[iterations - 2];
    System.out.println("Result:\n" + arraytoString(a.toArray(), 20));

    assertArrayEquals(A.toArray(), a.toArray(), 0.1);
  }


  @Test
  public void testDevDoubleAppend() {
    VecTools.append(vectors[0], vectors[1]);
    VecTools.append(vecs[0], vecs[1]);
    assertArrayEquals(vectors[0].toArray(), vecs[0].toArray(), DELTA);
  }


  @Test
  public void testDevDoubleInscale() {
    double scale = random.nextDouble();
    VecTools.incscale(vectors[0], vectors[1], scale);
    VecTools.incscale(vecs[0], vecs[1], scale);
    assertArrayEquals(vectors[0].toArray(), vecs[0].toArray(), DELTA);
  }


  @Test
  public void testDevDoubleScale() {
    double scale = random.nextDouble();
    VecTools.scale(vectors[0], scale);
    VecTools.scale(vecs[0], scale);
    assertArrayEquals(vectors[0].toArray(), vecs[0].toArray(), DELTA);
  }

  @Test
  public void testDevDoubleMultiply() {
    double multiply1 = VecTools.multiply(vectors[0], vectors[1]);
    double multiply2 = VecTools.multiply(vecs[0], vecs[1]);
    assertEquals(multiply1, multiply2, DELTA);
  }

  @Test
  public void testDevDoubleFill() {
    double value = random.nextDouble();
    VecTools.fill(vectors[0], value);
    VecTools.fill(vecs[0], value);
    assertArrayEquals(vectors[0].toArray(), vecs[0].toArray(), DELTA);
  }

  @Test
  public void testDevDoubleRightMult() {
    int rows = 20;
    int columns = SIZE / rows;

    Mx mx = new VecBasedMx(columns, vecs[0]);
    GPUMx matrix = new GPUMx(rows, vectors[0]);

    Vec vec = vecs[1].sub(0, columns);
    Vec vector = vectors[1].sub(0, columns);
    Vec multiply2 = new GPUVec(rows);

    Vec multiply1 = MxTools.multiply(mx, vec);
    multiply(matrix, vector, multiply2);
    assertArrayEquals(multiply1.toArray(), multiply2.toArray(), DELTA);
  }

  @Test
  public void testDevDoubleRightMultQ() {
    int rows = (int) Math.sqrt(SIZE);
    int columns = SIZE / rows;

    Mx mx = new VecBasedMx(columns, vecs[0]);
    GPUMx matrix = new GPUMx(rows, vectors[0]);

    Vec vec = vecs[1].sub(0, columns);
    Vec vector = vectors[1].sub(0, columns);
    //    Vec multiply2 = new gpuVec(rows);

    Vec multiply1 = MxTools.multiply(mx, vec);
    multiply(matrix, vector, vector);
    assertArrayEquals(multiply1.toArray(), vector.toArray(), DELTA);
  }

  public static void multiply(Mx mx, Vec vec, Vec result) {
    GPUMx mxResult = new GPUMx(mx.rows(), result);
    DeviceOperations.multiply((GPUMx) mx, new GPUMx(mx.columns(), vec), mxResult);
  }


  public static String arraytoString(double[] array, int limit) {
    StringBuilder builder = new StringBuilder("[");
    for (int i = 0; i < limit && i < array.length; i++) {
      builder.append(array[i]).append(",");
    }
    if (limit < array.length) {
      builder.append("...");
    } else {
      builder.deleteCharAt(builder.length() - 1);
    }
    builder.append("]");
    return builder.toString();
  }
}
