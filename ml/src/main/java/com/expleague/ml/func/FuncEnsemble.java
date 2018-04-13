package com.expleague.ml.func;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.Func;

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
    double weightTotal = 0;
    for (int i = 0; i < size(); i++) {
      double weight = weights.get(i);
      result += models[i].value(x) * weight;
      weightTotal += weight;
    }
    return result / weightTotal;
  }

  @Override
  public int dim() {
    return super.xdim();
  }
}
