package com.spbsu.ml.factorization.impl;

import com.spbsu.commons.math.vectors.*;
import com.spbsu.commons.math.vectors.impl.iterators.SkipVecNZIterator;
import com.spbsu.commons.math.vectors.impl.mx.ColsVecArrayMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.set.impl.VecDataSetImpl;
import com.spbsu.ml.factorization.OuterFactorization;
import com.spbsu.ml.func.Linear;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.ElasticNetMethod;

/**
 * User: qdeee
 * Date: 25.02.15
 */
public class ElasticNetFactorization implements OuterFactorization {
  private final int iters;
  private final double tolerance;
  private final double alpha;
  private final double lambda;

  public ElasticNetFactorization(final int iters, final double tolerance, final double alpha, final double lambda) {
    this.iters = iters;
    this.tolerance = tolerance;
    this.alpha = alpha;
    this.lambda = lambda;
  }

  @Override
  public Pair<Vec, Vec> factorize(final Mx X) {
    final ElasticNetMethod elasticNetMethod = new ElasticNetMethod(tolerance, alpha, lambda);

    final int m = X.rows();
    final int n = X.columns();

    final Vec currentU = VecTools.fill(new ArrayVec(X.rows()), 1.0);
    final Vec currentV = VecTools.fill(new ArrayVec(X.columns()), 1.0);
    final VecDataSet dsForUSearch = new VecDataSetImpl(createMxForDs(currentV, m), null);
    final VecDataSet dsForVSearch = new VecDataSetImpl(createMxForDs(currentU, n), null);
    final L2 targetForUSearch = new L2(X, dsForUSearch);
    final L2 targetForVSearch = new L2(MxTools.transpose(X), dsForVSearch);

    for (int iter = 0; iter < iters; iter++) {
      //fix v, search u:
      final Linear modelForU = (Linear) elasticNetMethod.fit(dsForUSearch, targetForUSearch);
      VecTools.assign(currentU, modelForU.weights);

      //fix u, search v:
      final Linear modelForV = (Linear) elasticNetMethod.fit(dsForVSearch, targetForVSearch);
      VecTools.assign(currentV, modelForV.weights);
    }
    return Pair.create(currentU, currentV);
  }

  private static Mx createMxForDs(final Vec vecToFix, final int columns) {
    final Vec[] vecs = new Vec[columns];
    for (int j = 0; j < columns; j++) {
      vecs[j] = arrangeVecInColumn(vecToFix, j * vecToFix.dim(), vecToFix.dim() * columns);
    }
    return new ColsVecArrayMx(vecs);
  }

  private static Vec arrangeVecInColumn(final Vec vecToArrange, final int startPos, final int totalLength) {
    return new Vec.Stub() {
      @Override
      public double get(final int i) {
        return (i >= startPos && i <= startPos + vecToArrange.dim())
            ? vecToArrange.get(i % vecToArrange.dim())
            : 0.;
      }

      @Override
      public int dim() {
        return totalLength;
      }

      @Override
      public VecIterator nonZeroes() {
        return new SkipVecNZIterator(this);
      }

      @Override
      public Vec sub(final int start, final int end) {
        throw new UnsupportedOperationException();
      }

      @Override
      public Vec set(final int i, final double val) {
        throw new UnsupportedOperationException();
      }

      @Override
      public Vec adjust(final int i, final double increment) {
        throw new UnsupportedOperationException();
      }
    };

  }
}
