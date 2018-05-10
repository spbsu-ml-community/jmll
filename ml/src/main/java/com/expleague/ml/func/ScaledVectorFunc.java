package com.expleague.ml.func;

import com.expleague.commons.math.DiscontinuousTrans;
import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import org.jetbrains.annotations.NotNull;

/**
 * User: qdeee
 * Date: 18.05.17
 */
public class ScaledVectorFunc extends Trans.Stub {
  public final Vec weights;
  public final Func function;

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
  public DiscontinuousTrans subgradient() {
    final DiscontinuousTrans subgradient = function.subgradient();
    if (subgradient == null) {
      throw new UnsupportedOperationException();
    }

    return new DiscontinuousTrans.Stub() {
      @NotNull
      @Override
      public Vec leftTo(Vec x, Vec to) {
        subgradient.leftTo(x, to);
        return VecTools.scale(to, VecTools.sum(weights));
      }

      @NotNull
      @Override
      public Vec rightTo(Vec x, Vec to) {
        subgradient.rightTo(x, to);
        return VecTools.scale(to, VecTools.sum(weights));
      }

      @Override
      public int xdim() {
        return ScaledVectorFunc.this.xdim();
      }

      @Override
      public int ydim() {
        return xdim();
      }
    };
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
