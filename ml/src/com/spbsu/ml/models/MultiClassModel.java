package com.spbsu.ml.models;

import com.spbsu.commons.func.Evaluator;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.Func;
import com.spbsu.ml.Trans;
import com.spbsu.ml.func.FuncJoin;

/**
 * User: solar
 * Date: 10.01.14
 * Time: 10:59
 */
public class MultiClassModel extends FuncJoin {
  public MultiClassModel(Func[] dirs) {
    super(dirs);
  }

  public double p(int classNo, Vec x) {
    final double value = ((Func)dirs[classNo]).value(x);
    return 1. / (1. + Math.exp(-value));
  }

  public int bestClass(final Vec x) {
    return ArrayTools.max(dirs, new Evaluator<Trans>() {
      @Override
      public double value(Trans trans) {
        return ((Func) trans).value(x);
      }
    });
  }
}
