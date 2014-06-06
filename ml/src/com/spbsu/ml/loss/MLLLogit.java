package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.FuncC1;

import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * We use value representation = \frac{e^x}{e^x + 1}.
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class MLLLogit extends FuncC1.Stub {
  private final Vec target;
  private final int classesCount;

  public MLLLogit(Vec target) {
    this.target = target;
    classesCount = (int)target.get(ArrayTools.max(target.toArray())) + 1;
  }

  @Override
  public Vec gradient(Vec point) {
    Vec result = new ArrayVec(point.dim());
    Mx resultMx = new VecBasedMx(target.dim(), result);
    Mx mxPoint = new VecBasedMx(target.dim(), point);
    for (int i = 0; i < target.dim(); i++) {
      double sum = 0;
      for (int c = 0; c < classesCount - 1; c++){
        double expX = exp(mxPoint.get(c, i));
        sum += expX;
      }
      final int pointClass = (int)target.get(i);
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
  public double value(Vec point) {
    double result = 0;
    Mx mxPoint = new VecBasedMx(target.dim(), point);
    for (int i = 0; i < target.dim(); i++) {
      double sum = 0;
      for (int c = 0; c < classesCount - 1; c++){
        double expX = exp(mxPoint.get(c, i));
        sum += expX;
      }
      final int pointClass = (int)target.get(i);
      if (pointClass != classesCount - 1)
        result += log(exp(mxPoint.get(pointClass, i)) / (1. + sum));
      else
        result += log(1./(1. + sum));
    }

    return exp(result / target.dim());
  }

  @Override
  public int dim() {
    return target.dim() * (classesCount - 1);
  }
}
