package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.Func;

public class QuadraticModel extends Func.Stub {
  private final Mx M;
  private final Vec b;
  private final double c;

  public QuadraticModel(final Mx m, final Vec b, final double c) {
    M = m;
    this.b = b;
    this.c = c;
  }

  @Override
  public int dim() {
    return b.dim();
  }

  @Override
  public double value(final Vec x) {
    return VecTools.multiply(x, MxTools.multiply(M, x)) + VecTools.multiply(x, b) + c;
  }

  @Override
  public String toString() {
    return M.toString() + "\n" + b.toString() + "\n" + c;
  }
}
