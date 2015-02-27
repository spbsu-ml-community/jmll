package com.spbsu;

import com.spbsu.ml.cuda.JcublasHelper;
import com.spbsu.ml.cuda.JcudaHelper;
import com.spbsu.ml.cuda.JcudaVectorInscale;
import com.spbsu.ml.cuda.data.FMatrix;
import com.spbsu.ml.cuda.data.impl.FArrayMatrix;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;

import java.util.Random;

/**
 * jmll
 * ksen
 * 21.February.2015 at 20:19
 */
public class Main {

  public static void main(final String[] args) {
    second();
  }

  private static void second() {
    for (int i = 1; i < 100; i++) {
      final float[] gpu = new float[i];
      for (int j = 0; j < i; j++) {
        gpu[j] = -1;
      }
      long begin = System.nanoTime();
      JcudaVectorInscale.fExp(gpu);
      System.out.println("Iter: " + i + "\tTime: GPU=" + (System.nanoTime() - begin));

      begin = System.nanoTime();
      JcudaVectorInscale.fSigmoid(gpu);
      System.out.println("Iter: " + i + "\tTime: GPU=" + (System.nanoTime() - begin));

      begin = System.nanoTime();
      JcudaVectorInscale.fTanh(gpu);
      System.out.println("Iter: " + i + "\tTime: GPU=" + (System.nanoTime() - begin));
    }

    JcudaHelper.destroy();
  }

  private static void first() {
    for (int i = 1; i < 5000; i += 100) {
      final Mx A = getMx(i);
      final Mx B = getMx(i);

      long cpuBegin = System.currentTimeMillis();
      final Mx C = MxTools.multiply(A, B);
      long cpuEnd = System.currentTimeMillis();

      final FMatrix D = getMatrix(i);
      final FMatrix E = getMatrix(i);

      long gpuBegin = System.currentTimeMillis();
      final FMatrix F = JcublasHelper.fSum(D, E);
      long gpuEnd = System.currentTimeMillis();

      System.out.println(
          "Dim: " + i + "\tCPU: " + (cpuEnd - cpuBegin) + "\tGPU: " + (gpuEnd - gpuBegin)
      );
    }
  }

  private static FMatrix getMatrix(final int dim) {
    final FMatrix A = new FArrayMatrix(dim, dim);
    final Random random = new Random();
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        A.set(i, j, random.nextFloat());
      }
    }
    return A;
  }

  private static Mx getMx(final int dim) {
    final Mx A = new VecBasedMx(dim, dim);
    final Random random = new Random();
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        A.set(i, j, random.nextDouble());
      }
    }
    return A;
  }

}
