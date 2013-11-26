package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.ml.FuncStub;

/**
 * User: solar
 * Date: 21.11.13
 * Time: 11:40
 */
public class LinearModel extends FuncStub {
  public final Vec weights;

  public LinearModel(Vec weights) {
    this.weights = weights;
  }

  @Override
  public int xdim() {
    return weights.dim();
  }

  @Override
  public double value(Vec x) {
    return VecTools.multiply(weights, x);
  }
}
