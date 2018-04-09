package com.expleague.ml.models.nn;

import com.expleague.commons.math.AnalyticFunc;

public class Identity extends AnalyticFunc.Stub {
  @Override
  public double value(double x) {
    return x;
  }

  @Override
  public double gradient(double x) {
    return 1.;
  }
}
