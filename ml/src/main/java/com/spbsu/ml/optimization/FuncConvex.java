package com.spbsu.ml.optimization;

import com.spbsu.ml.Func;
import com.spbsu.ml.FuncC1;

/**
 * User: qde
 * Date: 24.04.13
 * Time: 19:01
 */

public interface FuncConvex extends FuncC1 {
  double getGlobalConvexParam();
  double getGradLipParam();

  abstract class Stub extends Func.Stub implements FuncConvex {
    @Override
    public double getGlobalConvexParam() {
      return 1.;
    }
  }
}
