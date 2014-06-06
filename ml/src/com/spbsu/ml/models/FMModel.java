package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.*;
import com.spbsu.commons.math.vectors.impl.basis.IntBasis;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;
import com.spbsu.ml.FuncC1;

/**
 * User: qdeee
 * Date: 17.03.14
 */
public class FMModel extends FuncC1.Stub {
  private final Mx V;  //[k x p]
  private final Vec w;
  private final double w0;

  public FMModel(final Mx v, final Vec w, final double w0) {
    V = v;
    this.w = w;
    this.w0 = w0;
  }

  public FMModel(final Mx v, final Vec w) {
    this(v, w, 0.);
  }

  public FMModel(final Mx v) {
    this(v, new SparseVec<IntBasis>(new IntBasis(v.columns())), 0.);
  }

  public Mx getV() {
    return V;
  }

  public Vec getW() {
    return w;
  }

  public double getW0() {
    return w0;
  }

  @Override
  public Vec gradient(final Vec x) {
    return new ArrayVec(x.dim());
  }

  @Override
  public double value(final Vec x) {
    assert(x.dim() != V.columns());
    double value = w0 + VecTools.multiply(x, w);
    for (int k = 0; k < V.rows(); k++) {
      double sum = 0.;
      double sumSqr = 0.;
      VecIterator i = x.nonZeroes();
      final Vec row = V.row(k);
      while (i.advance()) {
        final double d = row.get(i.index()) * i.value();
        sum += d;
        sumSqr += d * d;
      }
      value += 0.5 * (sum * sum - sumSqr);
    }
    return value;
  }

  @Override
  public int dim() {
    return V.rows();
  }

  @Override
  public boolean equals(final Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;

    final FMModel fmModel = (FMModel) o;

    if (Double.compare(fmModel.w0, w0) != 0) return false;
    if (!V.equals(fmModel.V)) return false;
    if (!w.equals(fmModel.w)) return false;

    return true;
  }

  @Override
  public int hashCode() {
    int result;
    long temp;
    result = V.hashCode();
    result = 31 * result + w.hashCode();
    temp = Double.doubleToLongBits(w0);
    result = 31 * result + (int) (temp ^ (temp >>> 32));
    return result;
  }
}
