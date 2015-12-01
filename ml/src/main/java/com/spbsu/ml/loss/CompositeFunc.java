package com.spbsu.ml.loss;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ThreadLocalArrayVec;
import com.spbsu.commons.math.FuncC1;
import com.spbsu.commons.math.TransC1;

import static com.spbsu.commons.math.vectors.VecTools.assign;

/**
 * User: solar
 * Date: 01.06.15
 * Time: 13:52
 */
public class CompositeFunc extends FuncC1.Stub {
  private final FuncC1 first;
  private final TransC1[] after;

  public CompositeFunc(FuncC1 first, TransC1... after) {
    this.first = first;
    this.after = after;
    int dim = first.dim();
    for(int i = 0; i < after.length; i++) {
      if (dim != after[i].ydim())
        throw new IllegalArgumentException("Composite dimensions does not match: " + after[i] + " must have y dimension of " + dim);
      dim = after[i].xdim();
    }
  }

  final ThreadLocalArrayVec nextTemp = new ThreadLocalArrayVec();
  final ThreadLocalArrayVec gradTemp = new ThreadLocalArrayVec();
  final ThreadLocalArrayVec resultTemp = new ThreadLocalArrayVec();
  @Override
  public Vec gradient(Vec x) {
    final Vec[] values = new Vec[after.length + 1];
    values[after.length] = x;
    for (int i = after.length - 1; i >= 0; i--) {
      values[i] = after[i].trans(x);
      if (Double.isNaN(VecTools.norm(values[i]))) {
        throw new RuntimeException("" + after[i].trans(x));
      }
      x = values[i];
    }
    Vec result = first.gradientTo(x, resultTemp.get(first.xdim()));
    for (int i = 0; i < after.length; i++) {
      final Vec next = nextTemp.get(after[i].xdim());
      for (int j = 0; j < result.length(); j++) {
        if (Math.abs(result.get(j)) < MathTools.EPSILON)
          continue;
        final Vec grad = gradTemp.get(next.dim());
        final Vec gradientRowJ = after[i].gradientRowTo(values[i + 1], grad, j);
        VecTools.incscale(next, gradientRowJ, result.get(j));
      }
      result = resultTemp.get(next.dim());
      assign(result, next);
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
