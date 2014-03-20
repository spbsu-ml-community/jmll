package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.IntBasis;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.SparseVec;
import com.spbsu.ml.FuncC1;

/**
 * User: qdeee
 * Date: 17.03.14
 */
public class FactorizationMachineModel extends FuncC1.Stub {
  private final Mx V;  //[p x k]
  private final Vec w;
  private final double w0;

  public FactorizationMachineModel(final Mx v, final Vec w, final double w0) {
    V = v;
    this.w = w;
    this.w0 = w0;
  }

  public FactorizationMachineModel(final Mx v, final Vec w) {
    this(v, w, 0.);
  }

  public FactorizationMachineModel(final Mx v) {
    this(v, new SparseVec<IntBasis>(new IntBasis(v.rows())), 0.);
  }

  @Override
  public Vec gradient(final Vec x) {
    return null;
  }

  @Override
  public double value(final Vec x) {
    double value = w0 + VecTools.multiply(x, w);
    for (int k = 0; k < V.columns(); k++) {
      double sum = 0.;
      double sumSqr = 0.;
      for (int i = 0; i < x.dim(); i++) {
        final double d = V.get(i, k) * x.get(i);
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
}
