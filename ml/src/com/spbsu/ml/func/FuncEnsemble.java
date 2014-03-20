package com.spbsu.ml.func;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.Func;

import java.util.List;

/**
 * User: qdeee
 * Date: 17.03.14
 */
public class FuncEnsemble<F extends Func> extends Ensemble<F> implements Func{
  public FuncEnsemble(final F[] models, final Vec weights) {
    super(models, weights);
  }

  public FuncEnsemble(final List<F> weakModels, final double step) {
    super(weakModels, step);
  }

  @Override
  public double value(final Vec x) {
    return trans(x).get(0);
  }

  @Override
  public int dim() {
    return models[0].dim();
  }
}
