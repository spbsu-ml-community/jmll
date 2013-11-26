package com.spbsu.ml.func;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.ml.Func;
import com.spbsu.ml.FuncStub;

/**
 * User: solar
 * Date: 26.11.12
 * Time: 15:56
 */
public class Average<F extends Func> extends FuncStub {
  public final F[] models;

  public Average(F[] models) {
    this.models = models;
  }

  @Override
  public int xdim() {
    return models[0].xdim() * models.length;
  }

  public double value(Vec point) {
    final Mx mx = point instanceof Mx ? (Mx)point : new VecBasedMx(models[0].xdim(), point);
    double result = 0;
    for (int i = 0; i < mx.rows(); i++) {
      result += models[i].value(mx.row(i));
    }
    return result / mx.rows();
  }
}
