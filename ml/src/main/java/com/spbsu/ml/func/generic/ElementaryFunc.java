package com.spbsu.ml.func.generic;

import com.spbsu.commons.math.vectors.SingleValueVec;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.FuncC1;
import org.jetbrains.annotations.NotNull;

/**
 * User: solar
 * Date: 29.06.15
 * Time: 16:40
 */
public abstract class ElementaryFunc extends FuncC1.Stub {
  public abstract double value(double x);

  @NotNull
  public abstract ElementaryFunc gradient();

  @Override
  public final double value(Vec x) {
    return value(x.get(0));
  }

  @Override
  public final int dim() {
    return 1;
  }

  @Override
  public final Vec gradientTo(Vec x, Vec to) {
    to.set(0, gradient().value(x.get(0)));
    return to;
  }

  @Override
  public final Vec gradient(Vec x) {
    return new SingleValueVec(gradient().value(x.get(0)));
  }
}
