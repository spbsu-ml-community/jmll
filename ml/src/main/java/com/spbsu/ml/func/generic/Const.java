package com.spbsu.ml.func.generic;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.FuncC1;

/**
 * User: solar
 * Date: 10.06.15
 * Time: 23:40
 */
public class Const extends FuncC1.Stub {
  final double value;

  public Const(double value) {
    this.value = value;
  }

  @Override
  public double value(Vec x) {
    return value;
  }

  @Override
  public Vec gradientTo(Vec x, Vec to) {
    return to;
  }

  @Override
  public int dim() {
    return 0;
  }
}
