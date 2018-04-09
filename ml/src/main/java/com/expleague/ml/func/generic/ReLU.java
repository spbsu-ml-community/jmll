package com.expleague.ml.func.generic;

import com.expleague.commons.math.AnalyticFunc;

/**
* User: solar
* Date: 27.05.15
* Time: 17:59
*/
public class ReLU extends AnalyticFunc.Stub {
  @Override
  public double gradient(double x) {
    return x > 0 ? 1 : 0;
  }

  @Override
  public double value(double x) {
    return x > 0 ? x : 0;
  }
}
