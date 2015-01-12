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

/**
 * User: qdeee
 * Date: 12.01.15
 */
public class SVD implements OuterFactorization {
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

    final DenseMatrix64F huectorU = svd.getU(null, false);
    huectorU.reshape(huectorU.getNumRows(), 1);
    final Vec u = new ArrayVec(huectorU.getNumRows());
    for (int i = 0; i < huectorU.getNumRows(); i++) {
      u.set(i, huectorU.get(i, 0));
    }

    final double maxSingularValue = svd.getW(null).get(0, 0);
    VecTools.scale(u, maxSingularValue);

    final DenseMatrix64F huectorV = svd.getV(null, false);
    huectorV.reshape(1, huectorV.getNumCols());
    final Vec v = new ArrayVec(huectorV.getNumCols());
    for (int i = 0; i < huectorV.getNumCols(); i++) {
      v.set(i, huectorV.get(0, i));
    }

    return Pair.create(u, v);
  }
}
