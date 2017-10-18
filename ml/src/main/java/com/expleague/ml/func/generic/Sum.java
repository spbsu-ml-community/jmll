package com.expleague.ml.func.generic;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;

/**
* User: solar
* Date: 26.05.15
* Time: 11:45
*/
public class Sum extends FuncC1.Stub {
  @Override
  public Vec gradientTo(Vec x, Vec to) {
    VecTools.fill(to, 1.);
    return to;
  }

  @Override
  public double value(Vec x) {
    return VecTools.sum(x);
  }

  @Override
  public int dim() {
    return 1;
  }
}
