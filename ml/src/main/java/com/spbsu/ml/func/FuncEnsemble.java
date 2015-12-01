package com.spbsu.ml.func;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.Func;

import java.util.List;

/**
 * User: qdeee
 * Date: 09.04.14
 */

public class FuncEnsemble<X extends Func> extends Ensemble<X> implements Func{
  public FuncEnsemble(final X[] models, final Vec weights) {
    super(models, weights);
  }

  public FuncEnsemble(final List<X> models, final double step) {
    super(models, step);
  }

  @Override
  public double value(final Vec x) {
    double result = 0.;
    for (int i = 0; i < size(); i++) {
      result += models[i].value(x) * weights.get(i);
    }
    return result;
  }

  @Override
  public int dim() {
    return super.xdim();
  }
}
