package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.ml.Func;

public class QuadraticModel extends Func.Stub {
  private Mx M;
  private Vec b;
  private double c;

  public QuadraticModel(Mx m, Vec b, double c) {
    M = m;
    this.b = b;
    this.c = c;
  }

  @Override
  public int dim() {
    return b.dim();
  }

  @Override
  public double value(Vec x) {
    return VecTools.multiply(x, VecTools.multiply(M, x)) + VecTools.multiply(x, b) + c;
  }

  @Override
  public String toString() {
    return M.toString() + "\n" + b.toString() + "\n" + c;
  }
}
