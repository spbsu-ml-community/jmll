package com.spbsu.ml.func;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.Func;

import java.util.List;

/**
 * User: qdeee
 * Date: 09.04.14
 */

public class FuncEnsemble<X extends Func> extends Ensemble<X> implements Func{
  public FuncEnsemble(X[] models, Vec weights) {
    super(models, weights);
  }

  public FuncEnsemble(List<X> models, double step) {
    super(models, step);
  }

  public double value(Vec x) {
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
