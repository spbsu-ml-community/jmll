package com.spbsu.ml.optimization;

import com.spbsu.ml.Func;
import com.spbsu.ml.VecFunc;
import org.jetbrains.annotations.NotNull;

/**
 * User: qde
 * Date: 24.04.13
 * Time: 19:01
 */

public interface ConvexFunction extends Func {
  @NotNull
  VecFunc gradient();
  public double getGlobalConvexParam();
  public double getGradLipParam();
}
