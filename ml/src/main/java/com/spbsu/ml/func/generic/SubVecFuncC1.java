package com.spbsu.ml.func.generic;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.FuncC1;

/**
 * User: solar
 * Date: 23.06.15
 * Time: 17:53
 */
public class SubVecFuncC1 extends FuncC1.Stub {
  private final FuncC1 delegate;
  private final int start;
  private final int length;
  private final int dim;

  public SubVecFuncC1(FuncC1 delegate, int start, int length, int dim) {
    if (start + length > dim || start < 0)
      throw new ArrayIndexOutOfBoundsException();
    this.delegate = delegate;
    this.start = start;
    this.length = length;
    this.dim = dim;
  }

  @Override
  public Vec gradientTo(Vec x, Vec to) {
    return delegate.gradientTo(x.sub(start, length), to.sub(start, length));
  }

  @Override
  public double value(Vec x) {
    final Vec sub = x.sub(start, length);
    return delegate.value(sub);
  }

  @Override
  public int dim() {
    return dim;
  }
}
