package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.math.FuncC1;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.DataSet;

import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * We use value representation = \frac{e^x}{e^x + 1}.
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
@Deprecated
public class MLLLogit extends FuncC1.Stub implements TargetFunc {
  private final IntSeq target;
  private final DataSet<?> owner;
  private final int classesCount;

  public MLLLogit(final IntSeq target, final DataSet<?> owner) {
    this.target = target;
    this.owner = owner;
    classesCount = ArrayTools.max(target) + 1;
  }

  @Override
  public Vec gradient(final Vec point) {
    final Vec result = new ArrayVec(point.dim());
    final Mx resultMx = new VecBasedMx(target.length(), result);
    final Mx mxPoint = new VecBasedMx(target.length(), point);
    for (int i = 0; i < target.length(); i++) {
      double sum = 0;
      for (int c = 0; c < classesCount - 1; c++){
        final double expX = exp(mxPoint.get(c, i));
        sum += expX;
      }
      final int pointClass = target.at(i);
      for (int c = 0; c < classesCount - 1; c++){
        if (pointClass == c)
          resultMx.adjust(c, i, -(1. + sum - exp(mxPoint.get(c, i)))/(1. + sum));
        else
          resultMx.adjust(c, i, exp(mxPoint.get(c, i))/ (1. + sum));
      }
    }
    return result;
  }

  @Override
  public double value(final Vec point) {
    double result = 0;
    final Mx mxPoint = new VecBasedMx(target.length(), point);
    for (int i = 0; i < target.length(); i++) {
      double sum = 0;
      for (int c = 0; c < classesCount - 1; c++){
        final double expX = exp(mxPoint.get(c, i));
        sum += expX;
      }
      final int pointClass = target.at(i);
      if (pointClass != classesCount - 1)
        result += log(exp(mxPoint.get(pointClass, i)) / (1. + sum));
      else
        result += log(1./(1. + sum));
    }

    return exp(result / target.length());
  }

  @Override
  public int dim() {
    return target.length() * (classesCount - 1);
  }

  public int label(final int idx) {
    return target.at(idx);
  }

  public int classesCount() {
    return classesCount;
  }

  public IntSeq labels() {
    return target;
  }

  @Override
  public DataSet<?> owner() {
    return owner;
  }
}
