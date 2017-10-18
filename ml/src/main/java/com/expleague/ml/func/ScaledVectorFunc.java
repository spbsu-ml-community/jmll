package com.expleague.ml.func;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;

/**
 * User: qdeee
 * Date: 18.05.17
 */
public class ScaledVectorFunc extends Trans.Stub {
  private final Vec weights;
  private final Func function;

  public ScaledVectorFunc(Func function, Vec weights) {
    this.weights = weights;

    this.function = function;
  }

  @Override
  public Vec transTo(Vec argument, Vec to) {
    VecTools.assign(to, weights);
    VecTools.scale(to, function.value(argument));
    return to;
  }

  @Override
  public int xdim() {
    return function.xdim();
  }

  @Override
  public int ydim() {
    return weights.dim();
  }
}
