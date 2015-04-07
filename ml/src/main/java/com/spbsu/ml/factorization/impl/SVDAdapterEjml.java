package com.spbsu.ml.factorization.impl;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.factorization.OuterFactorization;
import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.DecompositionFactory;
import org.ejml.interfaces.decomposition.SingularValueDecomposition;
import org.ejml.ops.SingularOps;

/**
 * User: qdeee
 * Date: 12.01.15
 */
public class SVDAdapterEjml implements OuterFactorization {
  private final int factorDim;
  private final boolean needCompact;

  public SVDAdapterEjml(final int factorDim, final boolean needCompact) {
    this.factorDim = factorDim;
    this.needCompact = needCompact;
  }

  public SVDAdapterEjml(final int factorDim) {
    this(factorDim, true);
  }

  public SVDAdapterEjml() {
    this(1, true);
  }

  @Override
  public Pair<Vec, Vec> factorize(final Mx X) {
    final int m = X.rows();
    final int n = X.columns();

    final DenseMatrix64F denseMatrix64F = new DenseMatrix64F(m, n);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        denseMatrix64F.set(i, j, X.get(i, j));
      }
    }

    final SingularValueDecomposition<DenseMatrix64F> svd = DecompositionFactory.svd(m, n, true, true, needCompact);
    if (!DecompositionFactory.decomposeSafe(svd, denseMatrix64F)) {
      throw new IllegalStateException("Decomposition failed");
    }

    final DenseMatrix64F U = svd.getU(null, false);
    final DenseMatrix64F W = svd.getW(null);
    final DenseMatrix64F V = svd.getV(null, false);
    SingularOps.descendingOrder(U, false, W, V, false);

    if (W.getNumCols() < factorDim) {
      throw new IllegalStateException("Factor dim is too big for this mx. Try a smaller value (" + Math.min(W.getNumRows(), W.getNumCols()) + ") or disable compact svd mode by setting 'needCompact' = false");
    }

    final Mx u = getSubFromEjmlMatrix(U, 0, 0, m, factorDim);
    final Mx w = getSubFromEjmlMatrix(W, 0, 0, factorDim, factorDim);
    final Mx v = getSubFromEjmlMatrix(V, 0, 0, n, factorDim);

    final Vec mult = MxTools.multiply(u, w);
    return Pair.create(mult, (Vec) v);
  }

  private static Mx getSubFromEjmlMatrix(DenseMatrix64F ejmlMatrix, int iPos, int jPos, int height, int width) {
    final Mx result = new VecBasedMx(height, width);
    for (int i = iPos; i < height; i++) {
      for (int j = jPos; j < width; j++) {
        result.set(i, j, ejmlMatrix.get(i, j));
      }
    }
    return result;
  }
}
