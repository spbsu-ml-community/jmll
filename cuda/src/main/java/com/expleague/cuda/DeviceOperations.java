package com.expleague.cuda;


import com.expleague.cuda.data.GPUMx;
import com.expleague.cuda.data.GPUVec;
import jcuda.jcublas.JCublas;

/**
 * Created by hrundelb on 30.10.17.
 */
public class DeviceOperations {

  static {
    JCudaHelper.hook();
    JCublas.cublasInit();
  }

  // Operations on device in float

  public static void append(final GPUVec left, final GPUVec right) {
    JCublas.cublasSaxpy(left.length, 1.0f, right.devicePointer,
        1, left.devicePointer, 1);
  }


  public static void incscale(final GPUVec result, final GPUVec left, final float scale) {
    JCublas.cublasSaxpy(result.length, scale, left.devicePointer,
        1, result.devicePointer, 1);
  }

  public static void scale(final GPUVec left, final float scale) {
    JCublas.cublasSscal(left.length, scale, left.devicePointer, 1);
  }

  public static double multiply(final GPUVec left, final GPUVec right) {
    return JCublas.cublasSdot(left.length, left.devicePointer, 1, right.devicePointer, 1);
  }

  public static void multiply(final GPUMx left, final GPUMx right, final GPUMx result) {
    fMMmult(1.0f, right, false, left, false, 0.0f, result);
  }

  public static void multiplyTo(final GPUMx left, final GPUMx right, final GPUMx result) {
    fMMmult(1.0f, right, false, left, false, 1.0f, result);
  }

  private static void fMMmult(final float alpha, final GPUMx left, final boolean transposeLeft,
                              final GPUMx right, final boolean transposeRight, final float beta,
                              final GPUMx result) {

    final char opA = transposeLeft ? 'T' : 'N';
    final char opB = transposeRight ? 'T' : 'N';
    final int m = transposeLeft ? left.rows() : left.columns();
    final int n = transposeRight ? right.columns() : right.rows();
    final int k = transposeLeft ? left.columns() : left.rows();
    final int lda = left.columns();
    final int ldb = right.columns();
    final int ldc = transposeLeft ? left.rows() : left.columns();

    JCublas.cublasSgemm(opA, opB, m, n, k, alpha, left.gpuVec.devicePointer, lda, right.gpuVec
        .devicePointer, ldb, beta, result.gpuVec.devicePointer, ldc);
  }
}
