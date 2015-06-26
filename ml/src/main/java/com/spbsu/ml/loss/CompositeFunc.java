package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.FuncC1;
import com.spbsu.ml.Trans;

/**
 * User: solar
 * Date: 01.06.15
 * Time: 13:52
 */
public class CompositeFunc extends FuncC1.Stub {
  private final FuncC1 first;
  private final Trans[] after;

  public CompositeFunc(FuncC1 first, Trans... after) {
    this.first = first;
    this.after = after;
    int dim = first.dim();
    for(int i = 0; i < after.length; i++) {
      if (dim != after[i].ydim())
        throw new IllegalArgumentException("Composite dimensions does not match: " + after[i] + " must have y dimension of " + dim);
      dim = after[i].xdim();
    }
  }

  @Override
  public Vec gradient(Vec x) {
    final Vec[] values = new Vec[after.length + 1];
    values[after.length] = x;
    for (int i = after.length - 1; i >= 0; i--) {
      x = values[i] = after[i].trans(x);
    }
    Vec result = first.gradient(x);
    for (int i = 0; i < after.length; i++) {
      final Trans gradient = after[i].gradient();
      if (gradient == null)
        throw new RuntimeException("Transformation " + after[i] + " has no gradient!");
      final VecBasedMx prod = new VecBasedMx(1, result);
      final Vec nextGradValue = gradient.trans(values[i + 1]);

      final VecBasedMx nextGrad = new VecBasedMx(nextGradValue.dim() / prod.dim(), nextGradValue);
      result = MxTools.multiplyTo(prod, nextGrad,
              new VecBasedMx(1, i + 1 < after.length ? values[i + 1] : new ArrayVec(values[i + 1].dim())));
    }
    return result;
  }

  @Override
  public double value(Vec x) {
    for (int i = after.length - 1; i >= 0; i--) {
      x = after[i].trans(x);
    }
    return first.value(x);
  }

  @Override
  public int dim() {
    return after[after.length - 1].xdim();
  }

  public FuncC1 first() {
    return first;
  }

  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
    builder.append(first.toString());
    for(int i = 0; i < after.length; i++) {
      builder.append(" o ").append(after[i]);
    }
    return builder.toString();
  }
}
