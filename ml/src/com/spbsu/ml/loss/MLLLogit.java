package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.Oracle1;

import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * We use value representation = \frac{e^x}{e^x + 1}.
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class MLLLogit implements Oracle1 {
  private final Vec target;
  private final int classesCount;

  public MLLLogit(Vec target) {
    this.target = target;
    classesCount = ArrayTools.max(target.toArray()) + 1;
  }

  @Override
  public Vec gradient(Vec point) {
    Vec result = new ArrayVec(point.dim());
    for (int i = 0; i < point.dim(); i++) {
      double expX = exp(point.get(i));
      double pX = expX / (1 + expX);
      if (target.get(i) > 0) // positive example
        result.set(i, 1 - pX);
      else // negative
        result.set(i, -pX);
    }
    return result;
  }

  @Override
  public int dim() {
    return target.dim() * classesCount;
  }

  public double value(Vec point) {
    double result = 0;
    for (int i = 0; i < point.dim(); i++) {
      double expX = exp(point.get(i));
      double pX = expX / (1 + expX);
      if (target.get(i) > 0) // positive example
        result -= log(pX);
      else // negative
        result -= log(1 - pX);
    }

    return result;
  }
}
