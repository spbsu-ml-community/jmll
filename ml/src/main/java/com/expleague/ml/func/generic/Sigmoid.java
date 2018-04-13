package com.expleague.ml.func.generic;

import com.expleague.commons.math.AnalyticFunc;
import com.expleague.commons.math.MathTools;

/**
* User: solar
* Date: 26.05.15
* Time: 11:45
*/
public class Sigmoid extends AnalyticFunc.Stub {
  @Override
  public double gradient(double x) {
    final double exp = Math.exp(-x);
    return exp > 1/ MathTools.EPSILON ? 0 : exp / (1 + exp) / (1 + exp);
  }

  @Override
  public double value(double x) {
    return 1./(1. + Math.exp(-x));
  }
}
