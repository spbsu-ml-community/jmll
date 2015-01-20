package com.spbsu.ml.factorization.impl;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
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

    final SingularValueDecomposition<DenseMatrix64F> svd = DecompositionFactory.svd(m, n, true, true, true);
    if (!DecompositionFactory.decomposeSafe(svd, denseMatrix64F)) {
      throw new IllegalStateException("Decomposition failed");
    }

    final DenseMatrix64F U = svd.getU(null, false);
    final DenseMatrix64F W = svd.getW(null);
    final DenseMatrix64F V = svd.getV(null, false);
    SingularOps.descendingOrder(U, false, W, V, false);

    final Vec u = new ArrayVec(m);
    for (int i = 0; i < m; i++) {
      u.set(i, U.get(i, 0));
    }

    final double maxSingularValue = W.get(0, 0);
    VecTools.scale(u, maxSingularValue);

    final Vec v = new ArrayVec(n);
    for (int i = 0; i < n; i++) {
      v.set(i, V.get(i, 0));
    }

    return Pair.create(u, v);
  }
}
