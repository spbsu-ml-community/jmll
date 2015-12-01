package com.spbsu.ml.func.generic;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.TransC1;

/**
* User: solar
* Date: 26.05.15
* Time: 11:45
*/
public class ParallelFunc extends TransC1.Stub {
  private final int dim;
  private final ElementaryFunc func;

  public ParallelFunc(int dim, ElementaryFunc func) {
    this.dim = dim;
    this.func = func;
  }

  @Override
  public Vec transTo(Vec arg, Vec to) {
    for (int i = 0; i < arg.length(); i++) {
      to.set(i, func.value(arg.get(i)));
    }
    return to;
  }

  @Override
  public Vec gradientRowTo(Vec x, Vec to, int index) {
    to.set(index, func.gradient().value(x.get(index)));
    return to;
  }

  @Override
  public int xdim() {
    return dim;
  }

  @Override
  public int ydim() {
    return dim;
  }
}
