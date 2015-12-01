package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecIterator;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;
import com.spbsu.commons.math.FuncC1;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

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
    this(v, new SparseVec(v.columns()), 0.);
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
    throw new NotImplementedException();
  }

  @Override
  public double value(final Vec x) {
    assert(x.dim() != V.columns());
    double value = w0;
    final VecIterator iter = x.nonZeroes();
    while (iter.advance()) {
      value += iter.value() * w.get(iter.index());
    }

    for (int k = 0; k < V.rows(); k++) {
      double sum = 0.;
      double sumSqr = 0.;
      final VecIterator i = x.nonZeroes();
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
    return w.dim();
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
    final long temp;
    result = V.hashCode();
    result = 31 * result + w.hashCode();
    temp = Double.doubleToLongBits(w0);
    result = 31 * result + (int) (temp ^ (temp >>> 32));
    return result;
  }
}
