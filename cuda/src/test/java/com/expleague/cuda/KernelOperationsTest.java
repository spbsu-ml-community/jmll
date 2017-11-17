package com.expleague.cuda;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecIterator;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.cuda.data.GPUMx;
import com.expleague.cuda.data.GPUVec;
import org.apache.commons.math3.util.FastMath;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertArrayEquals;

/**
 * Created by hrundelb on 30.10.17.
 */
public class KernelOperationsTest {

  private static final int LENGTH = 10000;

  private static final float DELTA = 1e-4f; // (CUDA version) > 6.0 -> EPS = 1e-9

  //@Test
  public void testMatrixReduce() {
    final int states = 500;
    Random random = new FastRandom();
    final double[] doubles = new double[states * (states - 1)];
    for (int i = 0; i < doubles.length; i++) {
      doubles[i] = random.nextDouble();
    }
    GPUMx input = new GPUMx(states - 1, new GPUVec(doubles));
    GPUMx output = new GPUMx(states, states);
    System.out.println(DeviceOperationsTest.arraytoString(input.toArray(), 30));
    System.out.println(DeviceOperationsTest.arraytoString(output.toArray(), 30));
    KernelOperations.fMatrixReduce(input, output);
    System.out.println(DeviceOperationsTest.arraytoString(output.toArray(), 30));

    Mx weightMx = getWeightMx(new VecBasedMx(states - 1, new ArrayVec(doubles)), states);
    System.out.println(DeviceOperationsTest.arraytoString(weightMx.toArray(), 30));

    assertArrayEquals(output.toArray(), weightMx.toArray(), DELTA);
  }


  @Test
  public void testMatrixExp() {
    final int rows = 500;
    final int iterations = 1000;
    FastRandom random = new FastRandom();
    final double[] doubles = new double[rows * (rows - 1)];
    for (int i = 0; i < doubles.length; i++) {
      doubles[i] = random.nextDouble();
    }

    Vec params = new ArrayVec(doubles);
    Mx beta = new VecBasedMx(rows - 1, params);
    GPUMx gpuMx = new GPUMx(rows - 1, new GPUVec(doubles));
    GPUMx result = new GPUMx(rows, rows);

    System.out.println("On CPU:");
    long time = System.currentTimeMillis();
    Mx weightMx = null;
    for (int i = 0; i < iterations; i++) {
      weightMx = getWeightMx(beta, rows);
    }
    long diff = System.currentTimeMillis() - time;
    System.out.println("Time on CPU: " + diff + " ms");

    System.out.println("On GPU:");
    time = System.currentTimeMillis();
    for (int i = 0; i < iterations; i++) {
      KernelOperations.fMatrixExp(gpuMx, result);
    }
    diff = System.currentTimeMillis() - time;
    System.out.println("Time on GPU: " + diff + " ms");

    double[] results = result.toArray();
    double[] dResults = weightMx.toArray();
    System.out.println(DeviceOperationsTest.arraytoString(dResults, 30));
    System.out.println(DeviceOperationsTest.arraytoString(results, 30));
    assertArrayEquals(dResults, results, DELTA);
  }

  @Test
  public void testDFill() {
    final float[] expected = new float[LENGTH];
    final float[] actual = new float[LENGTH];

    final Random random = new Random();
    final float expectedValue = random.nextFloat();
    for (int i = 0; i < LENGTH; i++) {
      final float value = random.nextFloat();

      expected[i] = expectedValue;
      actual[i] = value;
    }
    final GPUVec a = new GPUVec(actual);

    KernelOperations.dFill(a, expectedValue);

    assertArrayEquals(expected, a.toFloatArray(), DELTA);
  }

  @Test
  public void testMatrixOperation1() {
    final int stateCount = 1000;
    final int iterations = 1000;

    Random random = new Random();
    final double weight = random.nextDouble();
    final double diff = random.nextDouble();
    final int to = random.nextInt(stateCount);
    final double[] distrs = new double[stateCount];
    final double[] expects = new double[stateCount];
    for (int i = 0; i < stateCount; i++) {
      distrs[i] = random.nextDouble();
      expects[i] = random.nextDouble();
    }

    final double[] weights = new double[stateCount * stateCount];
    for (int i = 0; i < stateCount * stateCount; i++) {
      weights[i] = random.nextDouble();
    }

    Vec distr1 = new ArrayVec(distrs);
    GPUVec distr2 = new GPUVec(distrs);
    Vec expected1 = new ArrayVec(expects);
    GPUVec expected2 = new GPUVec(expects);

    Mx wA1 = new VecBasedMx(stateCount, new ArrayVec(weights));
    GPUMx wA2 = new GPUMx(stateCount, new GPUVec(weights));

    Mx betaGrad1 = new VecBasedMx(stateCount - 1, new ArrayVec(new double[stateCount *
        (stateCount - 1)]));
    GPUMx betaGrad2 = new GPUMx(stateCount, new GPUVec(new double[stateCount *
        (stateCount - 1)]));

    System.out.println("Start CPU");
    long l = System.currentTimeMillis();
    for (int i = 0; i < iterations; i++) {
      betaGrad1 = getOperationOne(stateCount, weight, diff, distr1, expected1, betaGrad1, to, wA1);
    }
    Mx one = betaGrad1;
    long d = System.currentTimeMillis() - l;
    System.out.println("On CPU: " + d + " ms");
    System.out.println(DeviceOperationsTest.arraytoString(one.toArray(), 20));
    System.out.println("Start GPU");
    l = System.currentTimeMillis();
    for (int i = 0; i < iterations; i++) {
      KernelOperations.fMatrixKernel1((float)weight, (float)diff, distr2, expected2, betaGrad2, to, wA2);
    }
    d = System.currentTimeMillis() - l;
    System.out.println("On GPU: " + d + " ms");
    System.out.println(DeviceOperationsTest.arraytoString(betaGrad2.toArray(), 20));

    assertArrayEquals(one.toArray(), betaGrad2.toArray(), 0.01);
  }

  @Test
  public void testMatrixOperation2() {
    final int stateCount = 1000;
    final int iterations = 1000;

    Random random = new Random();
    final double lambda = random.nextDouble();
    final int to = random.nextInt(stateCount);

    final double[] weights = new double[stateCount * stateCount];
    for (int i = 0; i < stateCount * stateCount; i++) {
      weights[i] = random.nextDouble();
    }

    Mx wI1 = new VecBasedMx(stateCount, new ArrayVec(weights));
    GPUMx wI2 = new GPUMx(stateCount, new GPUVec(weights));

    Mx betaGrad1 = new VecBasedMx(stateCount - 1, new ArrayVec(new double[stateCount *
        (stateCount - 1)]));
    GPUMx betaGrad2 = new GPUMx(stateCount, new GPUVec(new double[stateCount *
        (stateCount - 1)]));

    System.out.println("Start CPU");
    long l = System.currentTimeMillis();
    for (int i = 0; i < iterations; i++) {
      betaGrad1 = getOperationTwo(stateCount, lambda, betaGrad1, to, wI1);
    }
    Mx one = betaGrad1;
    long d = System.currentTimeMillis() - l;
    System.out.println("On CPU: " + d + " ms");
    System.out.println(DeviceOperationsTest.arraytoString(one.toArray(), 20));
    System.out.println("Start GPU");
    l = System.currentTimeMillis();
    for (int i = 0; i < iterations; i++) {
      KernelOperations.fMatrixKernel2((float)lambda, betaGrad2, to, wI2);
    }
    d = System.currentTimeMillis() - l;
    System.out.println("On GPU: " + d + " ms");
    System.out.println(DeviceOperationsTest.arraytoString(betaGrad2.toArray(), 20));

    assertArrayEquals(one.toArray(), betaGrad2.toArray(), 0.01);
  }

  @Test
  public void testVectorKernel1() {
    int dim = 3000;
    Random random = new Random();
    double[] lastArr = new double[dim];
    double[] coorArr = new double[dim];
    double[] totalArr = new double[dim];
    for (int i = 0; i < lastArr.length; i++) {
      lastArr[i] = random.nextDouble() * 100;
      coorArr[i] = random.nextDouble() * 10;
      totalArr[i] = random.nextDouble() * 10;
    }
    double step = random.nextDouble();
    int sumSize = random.nextInt(10);

    Vec last = new ArrayVec(lastArr);
    Vec coor = new ArrayVec(coorArr);
    Vec total = new ArrayVec(totalArr);

    GPUVec dLast = new GPUVec(lastArr);
    GPUVec dCoor = new GPUVec(coorArr);
    GPUVec dTotal = new GPUVec(totalArr);

    Vec x = new ArrayVec(dim);
    GPUVec doubleX = new GPUVec(dim);

    Vec iterationAdjust = getIterationAdjust(last, coor, total, sumSize, step, x);
    System.out.println(DeviceOperationsTest.arraytoString(iterationAdjust.toArray(), 20));

    KernelOperations.fVectorKernel1(dLast, dCoor, dTotal, (float) step, sumSize, doubleX);
    System.out.println(DeviceOperationsTest.arraytoString(doubleX.toArray(), 20));

    assertArrayEquals(x.toArray(), doubleX.toArray(), DELTA);

  }

  private Mx getWeightMx(final Mx beta, final int stateCount) {
    final Mx w = new VecBasedMx(stateCount, stateCount);
    for (int i = 0; i < stateCount; i++) {
      double sum = 0;
      for (int j = 0; j < stateCount - 1; j++) {
        final double e = FastMath.exp(beta.get(i, j));
        sum += e;
        w.set(i, j, e);
      }
      w.set(i, stateCount - 1, 1);
      sum += 1;
      for (int j = 0; j < stateCount; j++) {
        w.set(i, j, w.get(i, j) / sum);
      }
    }

    return w;
  }


  private Mx getOperationOne(final int stateCount, final double weight, final double diff, final
  Vec distributionI, final Vec expectedValue, final Mx betaGradA,
                             final int to, final Mx wA) {
    for (int from = 0; from < stateCount; from++) {
      for (int j = 0; j < stateCount - 1; j++) {
        final double curW = wA.get(from, to);
        final double grad = 2 * weight * diff * distributionI.get(from) * expectedValue.get(to);
        if (j == to) {
          betaGradA.adjust(from, j, grad * curW * (1 - curW));
        }
        else {
          betaGradA.adjust(from, j, -grad * curW * wA.get(from, j));
        }
      }
    }
    return betaGradA;
  }

  private Mx getOperationTwo(final int stateCount, final double lambda, final Mx betaGradI,
                             final int to, final Mx wI) {
    for (int from = 0; from < stateCount; from++) {
      for (int j = 0; j < stateCount - 1; j++) {
        final double curW = wI.get(from, to);
        final double grad = lambda * wI.get(from, to);
        if (j == to) {
          betaGradI.adjust(from, j, grad * curW * (1 - curW));
        }
        else {
          betaGradI.adjust(from, j, -grad * curW * wI.get(from, j));
        }
      }
    }
    return betaGradI;
  }

  private Vec getIterationAdjust(Vec lastGrad, Vec gradCoordinateInverseFreq, Vec totalGrad,
                                 int size, double step, Vec x) {
    final VecIterator iterator = lastGrad.nonZeroes();
    while (iterator.advance()) {
      final int index = iterator.index();
      x.adjust(index, -step * gradCoordinateInverseFreq.get(index)
          * totalGrad.get(index) / size);
    }
    return x;
  }
}
