package com.spbsu.ml.cuda;

import com.spbsu.ml.cuda.data.impl.FloatMatrix;
import org.jetbrains.annotations.NotNull;

import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.mx.ColMajorArrayMx;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;

/**
 * Project jmll
 *
 * @author Ksen
 */
public class JCublasHelper { //todo(ksen): row-major support

  static {
    JCudaHelper.hook();
    JCublas.cublasInit();
  }

  public static void shutdown() {
    JCublas.cublasShutdown();
  }

  public static int max(
      final @NotNull ArrayVec a
  ) {
    return dMax(a.toArray());
  }

  public static int min(
      final @NotNull ArrayVec a
  ) {
    return dMin(a.toArray());
  }

  public static double dot(
      final @NotNull ArrayVec a,
      final @NotNull ArrayVec b
  ) {
    return dDot(a.toArray(), b.toArray());
  }

  public static double manhattan(
      final @NotNull ArrayVec a
  ) {
    return dManhattan(a.toArray());
  }

  public static double euclidean(  //todo(ksenon): failed
      final @NotNull ArrayVec a
  ) {
    throw new UnsupportedOperationException();
//    return dEuclidean(a.toArray());
  }

  public static ArrayVec scale(
      final double alpha,
      final @NotNull ArrayVec a
  ) {
    final double[] ha = a.toArray();
    dVscale(alpha, ha);
    return new ArrayVec(ha);
  }

  public static ArrayVec sum(
      final @NotNull ArrayVec a,
      final @NotNull ArrayVec b
  ) {
    return new ArrayVec(dVVsum(1, a.toArray(), b.toArray()));
  }

  public static ArrayVec subtr(
      final @NotNull ArrayVec a,
      final @NotNull ArrayVec b
  ) {
    return new ArrayVec(dVVsum(-1, a.toArray(), b.toArray()));
  }

  public static ArrayVec mult(
      final @NotNull ColMajorArrayMx A,
      final @NotNull ArrayVec b
  ) {
    return new ArrayVec(fMVmult(A.rows(), A.columns(), A.toArray(), false, b.toArray()));
  }

  public static ArrayVec mult(
      final @NotNull ArrayVec b,
      final @NotNull ColMajorArrayMx A
  ) {
    return new ArrayVec(fMVmult(A.rows(), A.columns(), A.toArray(), true, b.toArray()));
  }

  public static ColMajorArrayMx mult(
      final @NotNull ArrayVec a,
      final @NotNull ArrayVec b
  ) {
    return new ColMajorArrayMx(
        a.dim(),
        dMMmult(a.dim(), 1, b.dim(), 1, 1.f, a.toArray(), false, b.toArray(), false, 0.f, null)
    );
  }

  public static ColMajorArrayMx sum(
      final @NotNull ColMajorArrayMx A,
      final @NotNull ColMajorArrayMx B
  ) {
    return new ColMajorArrayMx(A.rows(), dVVsum(1, A.toArray(), B.toArray()));
  }

  public static ColMajorArrayMx subtr(
      final @NotNull ColMajorArrayMx A,
      final @NotNull ColMajorArrayMx B
  ) {
    return new ColMajorArrayMx(A.rows(), dVVsum(-1, B.toArray(), A.toArray()));
  }

  public static ColMajorArrayMx scale(
      final double alpha,
      final @NotNull ColMajorArrayMx A
  ) {
    dVscale(alpha, A.toArray());
    return A;
  }

  public static ColMajorArrayMx mult(
      final @NotNull ColMajorArrayMx A,
      final @NotNull ColMajorArrayMx B
  ) {
    return mult(A, false, B, false);
  }

  public static ColMajorArrayMx mult(
      final @NotNull ColMajorArrayMx A,
      final boolean transA,
      final @NotNull ColMajorArrayMx B,
      final boolean transB
  ) {
    final int rowsA = A.rows();
    final int columnsA = A.columns();
    return new ColMajorArrayMx(
        transA ? columnsA : rowsA,
        dMMmult(rowsA, columnsA, B.rows(), B.columns(), 1.f, A.toArray(), transA, B.toArray(), transB, 0.f, null)
    );
  }

  //--------------------------------------------------------------------------------------------------------------------

  /**
   * index(max(|a[n]|))
   *
   * */
  private static int dMax(final double[] ha) {
    final int n = ha.length;

    JCublas.cublasInit();

    final Pointer da = new Pointer();

    JCublas.cublasAlloc(n, Sizeof.DOUBLE, da);

    JCublas.cublasSetVector(n, Sizeof.DOUBLE, Pointer.to(ha), 1, da, 1);

    final int index = JCublas.cublasIdamax(n, da, 1);

    JCublas.cublasFree(da);

    JCublas.cublasShutdown();

    return index - 1;
  }

  /**
   * index(min(|a[n]|))
   *
   * */
  private static int dMin(final double[] ha) {
    final int n = ha.length;

    JCublas.cublasInit();

    final Pointer da = new Pointer();

    JCublas.cublasAlloc(n, Sizeof.DOUBLE, da);

    JCublas.cublasSetVector(n, Sizeof.DOUBLE, Pointer.to(ha), 1, da, 1);

    final int index = JCublas.cublasIdamin(n, da, 1);

    JCublas.cublasFree(da);

    JCublas.cublasShutdown();

    return index - 1;
  }

  /**
   * product = a[n] · b[n]
   *
   * */
  private static double dDot(final double[] ha, final double[] hb) {
    final int n = ha.length;

    JCublas.cublasInit();

    final Pointer da = new Pointer();
    final Pointer db = new Pointer();

    JCublas.cublasAlloc(n, Sizeof.DOUBLE, da);
    JCublas.cublasAlloc(n, Sizeof.DOUBLE, db);

    JCublas.cublasSetVector(n, Sizeof.DOUBLE, Pointer.to(ha), 1, da, 1);
    JCublas.cublasSetVector(n, Sizeof.DOUBLE, Pointer.to(hb), 1, db, 1);

    final double hc = JCublas.cublasDdot(n, da, 1, db, 1);

    JCublas.cublasFree(da);
    JCublas.cublasFree(db);

    JCublas.cublasShutdown();

    return hc;
  }

  /**
   * sum = sum(|a[n]|)
   *
   * */
  private static double dManhattan(final double[] ha) {
    final int n = ha.length;

    JCublas.cublasInit();

    final Pointer da = new Pointer();

    JCublas.cublasAlloc(n, Sizeof.DOUBLE, da);

    JCublas.cublasSetVector(n, Sizeof.DOUBLE, Pointer.to(ha), 1, da, 1);

    final double sum = JCublas.cublasDasum(n, da, 1);

    JCublas.cublasFree(da);

    JCublas.cublasShutdown();

    return sum;
  }

  /**
   * sum = sqrt(sum(a[n]^2))
   *
   * */
  private static double dEuclidean(final double[] ha) {
    final int n = ha.length;

    JCublas.cublasInit();

    final Pointer da = new Pointer();

    JCublas.cublasAlloc(n, Sizeof.DOUBLE, da);

    JCublas.cublasSetVector(n, Sizeof.DOUBLE, Pointer.to(ha), 1, da, 1);

    final double sum = JCublas.cublasDnrm2(n, da, 1);

    JCublas.cublasFree(da);

    JCublas.cublasShutdown();

    return sum;
  }

  /**
   * a[n] = alpha * a[n]
   *
   * */
  private static void dVscale(final double alpha, final double[] ha) {
    final int n = ha.length;

    JCublas.cublasInit();

    final Pointer da = new Pointer();

    JCublas.cublasAlloc(n, Sizeof.DOUBLE, da);

    JCublas.cublasSetVector(n, Sizeof.DOUBLE, Pointer.to(ha), 1, da, 1);

    JCublas.cublasDscal(n, alpha, da, 1);

    JCublas.cublasGetVector(n, Sizeof.DOUBLE, da, 1, Pointer.to(ha), 1);

    JCublas.cublasFree(da);

    JCublas.cublasShutdown();
  }

  /**
   * c[n] = alpha * a[n] + b[n]
   *
   * */
  private static double[] dVVsum(final double alpha, final double[] ha, final double[] hb) {
    final int n = ha.length;
    final double[] hc = new double[n];

    JCublas.cublasInit();

    final Pointer da = new Pointer();
    final Pointer db = new Pointer();

    JCublas.cublasAlloc(n, Sizeof.DOUBLE, da);
    JCublas.cublasAlloc(n, Sizeof.DOUBLE, db);

    JCublas.cublasSetVector(n, Sizeof.DOUBLE, Pointer.to(ha), 1, da, 1);
    JCublas.cublasSetVector(n, Sizeof.DOUBLE, Pointer.to(hb), 1, db, 1);

    JCublas.cublasDaxpy(n, alpha, da, 1, db, 1);

    JCublas.cublasGetVector(n, Sizeof.DOUBLE, db, 1, Pointer.to(hc), 1);

    JCublas.cublasFree(da);
    JCublas.cublasFree(db);

    JCublas.cublasShutdown();

    return hc;
  }

  /**
   * c[m] = alpha * op(A[m x n]) * b[n] + beta * c[m]
   *
   * */
  private static double[] fMVmult(int m, int n, final double[] hA, final boolean trans, final double[] hb) {
    final int mn = m * n;
    final char op = trans ? 't' : 'n';
    final double[] hc = new double[trans ? n : m];

    JCublas.cublasInit();

    final Pointer dA = new Pointer();
    final Pointer db = new Pointer();
    final Pointer dc = new Pointer();

    JCublas.cublasAlloc(mn, Sizeof.DOUBLE, dA);
    JCublas.cublasAlloc(trans ? m : n, Sizeof.DOUBLE, db);
    JCublas.cublasAlloc(trans ? n : m, Sizeof.DOUBLE, dc);

    JCublas.cublasSetVector(mn, Sizeof.DOUBLE, Pointer.to(hA), 1, dA, 1);
    JCublas.cublasSetVector(trans ? m : n, Sizeof.DOUBLE, Pointer.to(hb), 1, db, 1);
    JCublas.cublasSetVector(trans ? n : m, Sizeof.DOUBLE, Pointer.to(hc), 1, dc, 1);

    JCublas.cublasDgemv(op, m, n, 1, dA, m, db, 1, 0, dc, 1);

    JCublas.cublasGetVector(trans ? n : m, Sizeof.DOUBLE, dc, 1, Pointer.to(hc), 1);

    JCublas.cublasFree(dA);
    JCublas.cublasFree(db);
    JCublas.cublasFree(dc);

    JCublas.cublasShutdown();

    return hc;
  }

  /**
   * C[m x n] = alpha * op(A[m x k]) * op(B[k x n]) + beta * C[m x n]
   * */
  @SuppressWarnings("UnnecessaryLocalVariable")
  private static double[] dMMmult(
      final int rowsA,
      final int columnsA,
      final int rowsB,
      final int columnsB,
      final double alpha,
      final double[] hA,
      final boolean transA,
      final double[] hB,
      final boolean tranB,
      final double beta,
      double[] hC
  ) {
    final char opA = transA ? 'T' : 'N';
    final char opB = tranB ? 'T' : 'N';
    final int m = transA ? columnsA : rowsA;
    final int n = tranB ? rowsB : columnsB;
    final int k = transA ? rowsA : columnsA;
    final int lda = rowsA;
    final int ldb = rowsB;
    final int ldc = transA ? columnsA : rowsA;
    final int lengthA = hA.length;
    final int lengthB = hB.length;
    final int lengthC = m * n;

    hC = hC == null ? new double[lengthC] : hC;

    JCublas.cublasInit();

    final Pointer dA = new Pointer();
    final Pointer dB = new Pointer();
    final Pointer dC = new Pointer();

    JCublas.cublasAlloc(lengthA, Sizeof.DOUBLE, dA);
    JCublas.cublasAlloc(lengthB, Sizeof.DOUBLE, dB);
    JCublas.cublasAlloc(lengthC, Sizeof.DOUBLE, dC);

    JCublas.cublasSetVector(lengthA, Sizeof.DOUBLE, Pointer.to(hA), 1, dA, 1);
    JCublas.cublasSetVector(lengthB, Sizeof.DOUBLE, Pointer.to(hB), 1, dB, 1);
    JCublas.cublasSetVector(lengthC, Sizeof.DOUBLE, Pointer.to(hC), 1, dC, 1);

    JCublas.cublasDgemm(opA, opB, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc);

    JCublas.cublasGetVector(lengthC, Sizeof.DOUBLE, dC, 1, Pointer.to(hC), 1);

    JCublas.cublasFree(dA);
    JCublas.cublasFree(dB);
    JCublas.cublasFree(dC);

    JCublas.cublasShutdown();

    return hC;
  }

  // Operations on device

  public static void mult(
      final @NotNull FloatMatrix left,
      final @NotNull FloatMatrix right,
      final @NotNull FloatMatrix result
  ) {
    mult(left, false, right, false, result);
  }

  public static void mult(
      final @NotNull FloatMatrix left,
      final boolean transposeLeft,
      final @NotNull FloatMatrix right,
      final boolean transposeRight,
      final @NotNull FloatMatrix result
  ) {
    fMMmult(1.f, left, transposeLeft, right, transposeRight, 0.f, result);
  }

  @SuppressWarnings("UnnecessaryLocalVariable")
  private static void fMMmult(
      final float alpha,
      final FloatMatrix left,
      final boolean transposeLeft,
      final FloatMatrix right,
      final boolean transposeRight,
      final float beta,
      final @NotNull FloatMatrix result
  ) {
    final char opA = transposeLeft ? 'T' : 'N';
    final char opB = transposeRight ? 'T' : 'N';
    final int m = transposeLeft ? left.columns : left.rows;
    final int n = transposeRight ? right.rows : right.columns;
    final int k = transposeLeft ? left.rows : left.columns;
    final int lda = left.rows;
    final int ldb = right.rows;
    final int ldc = transposeLeft ? left.columns : left.rows;

    JCublas.cublasSgemm(
        opA, opB,
        m, n, k,
        alpha, left.devicePointer, lda,
        right.devicePointer, ldb,
        beta, result.devicePointer, ldc
    );
  }

}
