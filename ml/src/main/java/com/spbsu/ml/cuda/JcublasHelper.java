package com.spbsu.ml.cuda;

import org.jetbrains.annotations.NotNull;
import com.spbsu.ml.cuda.data.FMatrix;
import com.spbsu.ml.cuda.data.FVector;
import com.spbsu.ml.cuda.data.impl.FArrayMatrix;
import com.spbsu.ml.cuda.data.impl.FArrayVector;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;

/**
 * jmll
 * ksen
 * 14.October.2014 at 12:23
 */
public class JcublasHelper { //todo(ksen): row-major support

  static {
    JcudaHelper.warmUp();
  }

  public static FMatrix fMult(
      final @NotNull FMatrix A,
      final @NotNull FMatrix B
  ) {
    return fMult(A, false, B, false);
  }

  public static FMatrix fMult(
      final @NotNull FMatrix A,
      final boolean transA,
      final @NotNull FMatrix B,
      final boolean transB
  ) {
    final int rowsA = A.getRows();
    final int columnsA = A.getColumns();
    return new FArrayMatrix(
        transA ? columnsA : rowsA,
        fMMmult(rowsA, columnsA, B.getRows(), B.getColumns(), 1.f, A.toArray(), transA, B.toArray(), transB, 0.f, null)
    );
  }

  public static FVector fMult(
      final @NotNull FMatrix A,
      final @NotNull FVector b
  ) {
    return new FArrayVector(fMVmult(A.getRows(), A.getColumns(), A.toArray(), false, b.toArray()));
  }

  public static FVector fMult(
      final @NotNull FVector b,
      final @NotNull FMatrix A
  ) {
    return new FArrayVector(fMVmult(A.getRows(), A.getColumns(), A.toArray(), true, b.toArray()));
  }

  public static FMatrix fMult(
      final @NotNull FVector a,
      final @NotNull FVector b
  ) {
    return new FArrayMatrix(
        a.getDimension(),
        fMMmult(a.getDimension(), 1, b.getDimension(), 1, 1.f, a.toArray(), false, b.toArray(), false, 0.f, null)
    );
  }

  public static float fDot(
      final @NotNull FVector a,
      final @NotNull FVector b
  ) {
    return fDot(b.getDimension(), a.toArray(), b.toArray());
  }

  public static FMatrix fSum(
      final @NotNull FMatrix A,
      final @NotNull FMatrix B
  ) {
    return new FArrayMatrix(A.getRows(), fVVsum(A.toArray(), B.toArray(), 1.f));
  }

  public static FVector fSum(
      final @NotNull FVector a,
      final @NotNull FVector b
  ) {
    return new FArrayVector(fVVsum(a.toArray(), b.toArray(), 1.f));
  }

  public static FMatrix fSubtr(
      final @NotNull FMatrix A,
      final @NotNull FMatrix B
  ) {
    return new FArrayMatrix(A.getRows(), fVVsum(B.toArray(), A.toArray(), -1.f));
  }

  public static FVector fSubtr(
      final @NotNull FVector a,
      final @NotNull FVector b
  ) {
    return new FArrayVector(fVVsum(a.toArray(), b.toArray(), -1.f));
  }

  public static FMatrix fScale(
      final @NotNull FMatrix A,
      final float alpha
  ) {
    fVscale(A.toArray(), alpha);
    return A;
  }

  public static FVector fScale(
      final @NotNull FVector a,
      final float alpha
  ) {
    fVscale(a.toArray(), alpha);
    return a;
  }

  //--------------------------------------------------------------------------------------------------------------------

  private static float fDot(final int n, final float[] ha, final float[] hb) {
    JCublas.cublasInit();

    final Pointer da = new Pointer();
    final Pointer db = new Pointer();

    JCublas.cublasAlloc(n, Sizeof.FLOAT, da);
    JCublas.cublasAlloc(n, Sizeof.FLOAT, db);

    JCublas.cublasSetVector(n, Sizeof.FLOAT, Pointer.to(ha), 1, da, 1);
    JCublas.cublasSetVector(n, Sizeof.FLOAT, Pointer.to(hb), 1, db, 1);

    final float hc = JCublas.cublasSdot(n, da, 1, db, 1);

    JCublas.cublasFree(da);
    JCublas.cublasFree(db);

    JCublas.cublasShutdown();

    return hc;
  }

  private static float[] fVVsum(final float[] ha, final float[] hb, final float alpha) {
    final int n = ha.length;
    final float[] hc = new float[n];

    JCublas.cublasInit();

    final Pointer da = new Pointer();
    final Pointer db = new Pointer();

    JCublas.cublasAlloc(n, Sizeof.FLOAT, da);
    JCublas.cublasAlloc(n, Sizeof.FLOAT, db);

    JCublas.cublasSetVector(n, Sizeof.FLOAT, Pointer.to(ha), 1, da, 1);
    JCublas.cublasSetVector(n, Sizeof.FLOAT, Pointer.to(hb), 1, db, 1);

    JCublas.cublasSaxpy(n, alpha, da, 1, db, 1);

    JCublas.cublasGetVector(n, Sizeof.FLOAT, db, 1, Pointer.to(hc), 1);

    JCublas.cublasFree(da);
    JCublas.cublasFree(db);

    JCublas.cublasShutdown();

    return hc;
  }

  private static void fVscale(final float[] ha, final float alpha) {
    final int n = ha.length;

    JCublas.cublasInit();

    final Pointer da = new Pointer();

    JCublas.cublasAlloc(n, Sizeof.FLOAT, da);

    JCublas.cublasSetVector(n, Sizeof.FLOAT, Pointer.to(ha), 1, da, 1);

    JCublas.cublasSscal(n, alpha, da, 1);

    JCublas.cublasGetVector(n, Sizeof.FLOAT, da, 1, Pointer.to(ha), 1);

    JCublas.cublasFree(da);

    JCublas.cublasShutdown();
  }

  private static float[] fMVmult(int m, int n, final float[] hA, final boolean trans, final float[] hb) {
    final int mn = m * n;
    final char op = trans ? 't' : 'n';
    final float[] hc = new float[trans ? n : m];

    JCublas.cublasInit();

    final Pointer dA = new Pointer();
    final Pointer db = new Pointer();
    final Pointer dc = new Pointer();

    JCublas.cublasAlloc(mn, Sizeof.FLOAT, dA);
    JCublas.cublasAlloc(trans ? m : n, Sizeof.FLOAT, db);
    JCublas.cublasAlloc(trans ? n : m, Sizeof.FLOAT, dc);

    JCublas.cublasSetVector(mn, Sizeof.FLOAT, Pointer.to(hA), 1, dA, 1);
    JCublas.cublasSetVector(trans ? m : n, Sizeof.FLOAT, Pointer.to(hb), 1, db, 1);
    JCublas.cublasSetVector(trans ? n : m, Sizeof.FLOAT, Pointer.to(hc), 1, dc, 1);

    JCublas.cublasSgemv(op, m, n, 1.f, dA, m, db, 1, 0.f, dc, 1);

    JCublas.cublasGetVector(trans ? n : m, Sizeof.FLOAT, dc, 1, Pointer.to(hc), 1);

    JCublas.cublasFree(dA);
    JCublas.cublasFree(db);
    JCublas.cublasFree(dc);

    JCublas.cublasShutdown();

    return hc;
  }

  @SuppressWarnings("UnnecessaryLocalVariable")
  private static float[] fMMmult(
      final int rowsA,
      final int columnsA,
      final int rowsB,
      final int columnsB,
      final float alpha,
      final float[] hA,
      final boolean transA,
      final float[] hB,
      final boolean tranB,
      final float beta,
      float[] hC
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

    hC = hC == null ? new float[lengthC] : hC;

    JCublas.cublasInit();

    final Pointer dA = new Pointer();
    final Pointer dB = new Pointer();
    final Pointer dC = new Pointer();

    JCublas.cublasAlloc(lengthA, Sizeof.FLOAT, dA);
    JCublas.cublasAlloc(lengthB, Sizeof.FLOAT, dB);
    JCublas.cublasAlloc(lengthC, Sizeof.FLOAT, dC);

    JCublas.cublasSetVector(lengthA, Sizeof.FLOAT, Pointer.to(hA), 1, dA, 1);
    JCublas.cublasSetVector(lengthB, Sizeof.FLOAT, Pointer.to(hB), 1, dB, 1);
    JCublas.cublasSetVector(lengthC, Sizeof.FLOAT, Pointer.to(hC), 1, dC, 1);

    JCublas.cublasSgemm(opA, opB, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc);

    JCublas.cublasGetVector(lengthC, Sizeof.FLOAT, dC, 1, Pointer.to(hC), 1);

    JCublas.cublasFree(dA);
    JCublas.cublasFree(dB);
    JCublas.cublasFree(dC);

    JCublas.cublasShutdown();

    return hC;
  }

}
